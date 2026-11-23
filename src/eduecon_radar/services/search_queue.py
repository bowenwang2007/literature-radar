"""Durable, local work queue for searches explicitly submitted in the workbench.

The queue is a sibling of its paper database. Provider keys never enter the queue.
A separate SQLite writer lock is released by the OS on process exit. Checkpoints
remain readable while a provider call is running.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, date, datetime

from eduecon_radar.config import Settings
from eduecon_radar.db import Database
from eduecon_radar.services.crossref import CrossrefClient
from eduecon_radar.services.deepseek import BATCH_REQUEST_LIMIT, limit_batch_requests
from eduecon_radar.services.journal_registry import RegistryJournal
from eduecon_radar.services.journal_sync import JournalSyncResult, JournalSyncService
from eduecon_radar.services.local_pipeline import (
    ensure_paper_statuses,
    process_batch,
    translate_titles_for_paper_ids,
)
from eduecon_radar.services.pipeline_builder import build_local_pipeline_services
from eduecon_radar.services.profile_registry import load_profile_registry

LOG = logging.getLogger(__name__)
_threads = {}
_guard = threading.Lock()
_wake = threading.Event()


def _combine_attempts(first: JournalSyncResult, last: JournalSyncResult) -> JournalSyncResult:
    """Keep every matched paper, even when a different page fails on the retry."""
    for field in (
        "paper_ids",
        "matched_paper_ids",
        "inserted_paper_ids",
        "updated_paper_ids",
        "unchanged_paper_ids",
        "duplicate_paper_ids",
    ):
        setattr(last, field, list(dict.fromkeys([*getattr(first, field), *getattr(last, field)])))
    inserted = set(last.inserted_paper_ids)
    last.updated_paper_ids = [i for i in last.updated_paper_ids if i not in inserted]
    changed = inserted.union(last.updated_paper_ids)
    last.unchanged_paper_ids = [i for i in last.unchanged_paper_ids if i not in changed]
    last.inserted = len(last.inserted_paper_ids)
    last.updated = len(last.updated_paper_ids)
    last.unchanged = len(last.unchanged_paper_ids)
    for field in (
        "pages_requested",
        "raw_candidate_count",
        "provider_returned_count",
        "normalized_count",
        "found_count",
        "found",
    ):
        setattr(last, field, getattr(first, field) + getattr(last, field))
    return last


class SearchQueue:
    def __init__(self, database: Database):
        self.db = database
        self.path = database.path.with_suffix(database.path.suffix + ".jobs.sqlite")

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY, profile TEXT NOT NULL, request TEXT NOT NULL,
                started TEXT NOT NULL UNIQUE, phase TEXT NOT NULL DEFAULT 'fetch',
                run_id INTEGER, due REAL NOT NULL DEFAULT 0, failures INTEGER DEFAULT 0,
                progress TEXT DEFAULT '等待检索', checkpoints TEXT DEFAULT '{}')""")
        return conn

    def submit(self, profile_id, journals, since, until, analyze, *, series=None, retry_after=0):
        request = json.dumps(
            {
                "profile": profile_id,
                "journals": [asdict(j) for j in journals],
                "since": since.isoformat(),
                "until": until.isoformat(),
                "analyze": analyze,
                **({"series": series} if series else {}),
            },
            sort_keys=True,
        )
        with closing(self.connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT id FROM jobs WHERE request=? AND phase!='done' AND phase!='abstracts'",
                (request,)
            ).fetchone()
            if series:
                existing = conn.execute(
                    "SELECT id FROM jobs WHERE profile=? AND json_extract(request,'$.series')=? "
                    "AND (phase!='done' OR (json_extract(request,'$.until')=? AND id>?)) "
                    "ORDER BY id DESC",
                    (profile_id, series, until.isoformat(), retry_after),
                ).fetchone()
            if existing:
                return existing["id"]
            cursor = conn.execute(
                "INSERT INTO jobs(profile,request,started) VALUES(?,?,?)",
                (profile_id, request, datetime.now(UTC).isoformat()),
            )
            _wake.set()
            return cursor.lastrowid

    def retrieval_request(self, profile_id, run_id):
        """Retrieve the journals and dates actually submitted, including custom journals."""
        with closing(self.connect()) as conn:
            row = conn.execute(
                "SELECT request FROM jobs WHERE profile=? AND run_id=? ORDER BY id DESC LIMIT 1",
                (profile_id, run_id),
            ).fetchone()
        return json.loads(row["request"]) if row else None

    def latest(self, profile):
        if not self.path.exists():
            return None
        with closing(self.connect()) as conn:
            row = conn.execute(
                "SELECT id,phase,progress,run_id FROM jobs WHERE profile=? "
                "ORDER BY id DESC LIMIT 1",
                (profile,),
            ).fetchone()
            return dict(row) if row else None

    def status(self, job_id):
        with closing(self.connect()) as conn:
            row = conn.execute(
                "SELECT id,phase,progress,run_id FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def active(self, profile):
        if not self.path.exists():
            return []
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT id,phase,progress,run_id FROM jobs WHERE profile=? AND phase!='done' "
                "ORDER BY id DESC",
                (profile,),
            ).fetchall()
        return [dict(row) for row in rows]

    def paused(self, job_id):
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT phase FROM jobs WHERE id=?", (job_id,)).fetchone()
        return row["phase"].startswith("paused:")

    def pause(self, job_id):
        with closing(self.connect()) as conn, conn:
            conn.execute(
                "UPDATE jobs SET phase='paused:'||phase WHERE id=? "
                "AND phase IN ('fetch','analysis','abstracts')",
                (job_id,),
            )

    def update(self, job, **values):
        assignments = ",".join(f"{field}=?" for field in values)
        with closing(self.connect()) as conn, conn:
            conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id=? AND phase NOT LIKE 'paused:%'",
                (*values.values(), job),
            )

    def checkpoint(self, job_id, phase, state):
        # An in-flight source request can finish after Pause was clicked. Save its
        # result without changing the user's requested phase or starting more work.
        with closing(self.connect()) as conn, conn:
            conn.execute(
                "UPDATE jobs SET checkpoints=? WHERE id=? AND phase IN (?,?)",
                (json.dumps(state), job_id, phase, "paused:" + phase),
            )

    def restart(self, job_id):
        with closing(self.connect()) as conn, conn:
            conn.execute(
                "UPDATE jobs SET phase=substr(phase,instr(phase,':')+1),due=0,failures=0 "
                "WHERE id=? AND (phase LIKE 'paused:%' OR phase LIKE 'attention:%')",
                (job_id,),
            )
        _wake.set()

    def start(self, settings):
        if not self.path.exists():
            return
        with _guard:
            thread = _threads.get(self.path)
            if thread is not None and thread.is_alive():
                return
            with closing(self.connect()) as conn:
                if not conn.execute(
                    "SELECT 1 FROM jobs WHERE phase IN ('fetch','analysis','abstracts') LIMIT 1"
                ).fetchone():
                    return
            thread = threading.Thread(target=self.work, args=(settings,), daemon=True)
            _threads[self.path] = thread
            thread.start()

    def work(self, settings: Settings):
        with closing(sqlite3.connect(str(self.path) + ".lock", timeout=0)) as lock:
            try:
                lock.execute("BEGIN EXCLUSIVE")
            except sqlite3.OperationalError:
                return
            while True:
                with closing(self.connect()) as conn:
                    job = conn.execute(
                        "SELECT * FROM jobs WHERE phase IN ('fetch','analysis','abstracts') "
                        "ORDER BY due,id LIMIT 1"
                    ).fetchone()
                if not job:
                    # Serialize idle exit with start(): a new submit must always have a worker.
                    with _guard, closing(self.connect()) as conn:
                        if conn.execute(
                            "SELECT 1 FROM jobs WHERE phase IN ('fetch','analysis','abstracts')"
                        ).fetchone():
                            continue
                        if _threads.get(self.path) is threading.current_thread():
                            _threads.pop(self.path)
                    return
                delay = job["due"] - time.time()
                if delay > 0:
                    _wake.wait(min(delay, 60))
                    _wake.clear()
                    continue
                try:
                    self.step(dict(job), settings)
                except Exception:
                    LOG.exception("Search job %s interrupted", job["id"])
                    failures = job["failures"] + 1
                    self.update(
                        job["id"],
                        failures=failures,
                        due=time.time() + 30 * failures,
                        phase=f"attention:{job['phase']}" if failures >= 3 else job["phase"],
                        progress="已保留进度，可继续尝试"
                        if failures >= 3
                        else "连接中断，稍后自动继续",
                    )

    def _budget(self, job_id, *, reserve=False):
        with closing(self.connect()) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS request_usage "
                         "(job INTEGER PRIMARY KEY, used INTEGER NOT NULL)")
            if reserve:
                return bool(conn.execute(
                    "INSERT INTO request_usage VALUES(?,1) ON CONFLICT(job) DO UPDATE "
                    "SET used=used+1 WHERE used<?", (job_id, BATCH_REQUEST_LIMIT)
                ).rowcount)
            row = conn.execute("SELECT used FROM request_usage WHERE job=?", (job_id,)).fetchone()
            return not row or row[0] < BATCH_REQUEST_LIMIT

    def step(self, job, settings):
        if job["phase"] == "fetch":
            self.fetch(job, json.loads(job["request"]))
            return
        run_id = job["run_id"]
        with self.db.connect() as conn:
            run = conn.execute("SELECT started_at FROM sync_runs WHERE id=?", (run_id,)).fetchone()
        if not run or run["started_at"] != job["started"]:
            self.update(job["id"], phase="done", progress="原检索记录已变更，后台任务已结束")
            return
        with limit_batch_requests(lambda: self._budget(job["id"], reserve=True)):
            if job["phase"] == "abstracts":
                self._recheck_abstract(job, settings)
            elif not self._budget(job["id"]):
                self._finish_analysis(job, "已达本批 AI 请求上限，剩余论文可按需处理")
            else:
                self._analyze(job, settings)

    def _finish_analysis(self, job, summary):
        with self.db.connect() as conn:
            missing = [row[0] for row in conn.execute(
                "SELECT p.id FROM paper_processing_status ps JOIN papers p ON p.id=ps.paper_id "
                "WHERE ps.sync_run_id=? AND length(trim(coalesce(p.abstract,'')))=0 "
                "AND NOT (ps.stage='complete' AND ps.status='classified') ORDER BY p.id",
                (job["run_id"],),
            )]
        if missing:
            self.update(
                job["id"], phase="abstracts", due=time.time()+86400,
                checkpoints=json.dumps({"missing": missing}),
                progress=f"{summary} · {len(missing)} 篇次日再查公开摘要",
            )
        else:
            self.update(job["id"], phase="done", progress=summary)

    def _recheck_abstract(self, job, settings):
        from eduecon_radar.services.pipeline_builder import build_enrichment_service

        state = json.loads(job["checkpoints"])
        pending = state["missing"]
        if not pending:
            summary = "摘要复查已结束"
            for key, label in (("missing_count", "仍缺摘要"), ("pending_ai", "已补摘要，待分析"),
                               ("failed_count", "复查失败")):
                if state.get(key):
                    summary += f" · {label} {state[key]} 篇"
            self.update(job["id"], phase="done", progress=summary, checkpoints="{}")
            return
        paper = pending[0]
        self.update(job["id"], progress=f"正在复查公开摘要 · 剩余 {len(pending)} 篇")
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT p.abstract FROM papers p JOIN paper_processing_status ps "
                "ON ps.paper_id=p.id "
                "WHERE p.id=? AND ps.sync_run_id=? AND NOT (ps.status='classified' "
                "AND ps.stage='complete')", (paper, job["run_id"]),
            ).fetchone()
        if row:
            try:
                if not (row[0] or "").strip():
                    build_enrichment_service(self.db, settings).enrich_papers(
                        (paper,), workflow_batch_id=None, force_refresh=True
                    )
                    with self.db.connect() as conn:
                        row = conn.execute(
                            "SELECT abstract FROM papers WHERE id=?", (paper,)
                        ).fetchone()
                if row and (row[0] or "").strip():
                    if settings.deepseek_api_key and self._budget(job["id"]):
                        profile = load_profile_registry(
                            settings.profiles_root, catalog_path=settings.journal_catalog_path,
                        ).profile(job["profile"])
                        services = build_local_pipeline_services(self.db, settings, profile=profile)
                        result = process_batch(self.db, job["run_id"], (paper,), services)
                        if not result["completed"]:
                            state["pending_ai"] = state.get("pending_ai", 0) + 1
                    else:
                        with self.db.connect() as conn:
                            conn.execute(
                                "UPDATE paper_processing_status SET status='abstract_ready', "
                                "stage='abstract_translation', error_type=NULL,error_message=NULL, "
                                "updated_at=? WHERE sync_run_id=? AND paper_id=? AND "
                                "NOT (status='classified' AND stage='complete')",
                                (datetime.now(UTC).isoformat(), job["run_id"], paper),
                            )
                        state["pending_ai"] = state.get("pending_ai", 0) + 1
                elif row:
                    state["missing_count"] = state.get("missing_count", 0) + 1
            except Exception:
                LOG.exception("Abstract follow-up interrupted for paper %s", paper)
                if paper not in state.setdefault("retried", []):
                    pending.append(paper)
                    state["retried"].append(paper)
                else:
                    state["failed_count"] = state.get("failed_count", 0) + 1
        state["missing"] = pending[1:]
        self.checkpoint(job["id"], "abstracts", state)
        self.update(job["id"], failures=0, due=time.time(),
                    progress=f"正在复查公开摘要 · 剩余 {len(state['missing'])} 篇")

    def _analyze(self, job, settings):
        run_id = job["run_id"]
        if not settings.deepseek_api_key:
            self.update(
                job["id"], phase=f"attention:{job['phase']}", progress="配置 AI 后可继续分析"
            )
            return
        profile = load_profile_registry(
            settings.profiles_root,
            catalog_path=settings.journal_catalog_path,
        ).profile(job["profile"])
        services = build_local_pipeline_services(self.db, settings, profile=profile)
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT ps.paper_id,ps.status,ps.stage,ps.attempts,p.abstract "
                "FROM paper_processing_status ps "
                "JOIN papers p ON p.id=ps.paper_id WHERE ps.sync_run_id=? ORDER BY ps.paper_id",
                (run_id,),
            ).fetchall()
        unfinished = [r for r in rows if r["status"] not in ("classified", "needs_manual_abstract")]
        fresh = [r["paper_id"] for r in unfinished if r["attempts"] == 0]
        retry = [
            r["paper_id"]
            for r in unfinished
            if r["attempts"] < 2
            or r["status"] in {"processing", "abstract_ready", "classification_ready"}
        ]
        attempts = {r["paper_id"]: r["attempts"] for r in unfinished}
        batch = (fresh or retry)[:10]
        if batch:
            self.update(
                job["id"],
                progress=f"正在整理文献 · 已处理 {len(rows) - len(unfinished)}/{len(rows)} 篇",
            )
            if not fresh and not json.loads(job["checkpoints"]).get("retry_started"):
                self.update(
                    job["id"],
                    due=time.time() + 30,
                    checkpoints=json.dumps({"retry_started": True}),
                    progress="部分请求暂时失败，稍后自动重试",
                )
                return
            translate_titles_for_paper_ids(self.db, run_id, tuple(batch), services)
            for paper_id in batch:
                if self.paused(job["id"]):
                    return
                try:
                    process_batch(self.db, run_id, (paper_id,), services)
                except Exception:
                    LOG.exception("Paper %s needs another attempt", paper_id)
                    with self.db.connect() as conn:
                        conn.execute(
                            "UPDATE paper_processing_status SET status='needs_retry', "
                            "attempts=max(attempts,?) "
                            "WHERE sync_run_id=? AND paper_id=?",
                            (attempts[paper_id] + 1, run_id, paper_id),
                        )
            self.update(job["id"], failures=0)
            return
        summary = f"本批 {len(rows)} 篇，可逐篇审核"
        if unfinished:
            summary += f" · {len(unfinished)} 篇分析仍待重试"
        self._finish_analysis(job, summary)

    def fetch(self, job, request):
        journals = tuple(RegistryJournal(**j) for j in request["journals"])
        since, until = date.fromisoformat(request["since"]), date.fromisoformat(request["until"])
        checkpoints = json.loads(job["checkpoints"])
        client = CrossrefClient()
        sync = JournalSyncService(self.db, client, profile_id=job["profile"])
        try:
            for journal in journals:
                if self.paused(job["id"]):
                    return
                previous = checkpoints.get(journal.journal_key)
                if previous and (not previous.get("query_errors") or previous.get("_retried")):
                    continue
                self.update(job["id"], progress=f"正在检索 {journal.canonical_name}")
                (result,) = sync.collect(
                    (journal,), since=since, until=until, per_journal_limit=None
                )
                if previous:
                    result = _combine_attempts(JournalSyncResult(**previous), result)
                checkpoints[journal.journal_key] = asdict(result)
                if previous:
                    checkpoints[journal.journal_key]["_retried"] = True
                self.checkpoint(job["id"], "fetch", checkpoints)
        finally:
            client.client.close()
        if any(row["query_errors"] and not row.get("_retried") for row in checkpoints.values()):
            self.update(
                job["id"], due=time.time() + 30, progress="已保留找到的论文，稍后重查异常期刊"
            )
            return
        for row in checkpoints.values():
            row.pop("_retried", None)
        results = [JournalSyncResult(**row) for row in checkpoints.values()]
        run_id = sync.record(journals, since, until, None, results, started_at=job["started"])
        ids = ensure_paper_statuses(self.db, run_id)
        analyze = request["analyze"] and bool(ids)
        partial = any(r.query_errors for r in results)
        self.update(
            job["id"],
            run_id=run_id,
            checkpoints="{}",
            failures=0,
            phase="analysis" if analyze else "done",
            progress=f"已获取 {len(ids)} 篇" + (" · 部分来源返回不完整" if partial else ""),
        )
