"""Overnight campaign export pack.

Produces the seven required output files scoped to the real-run papers plus the
task #17 papers. Exports never write to the database.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

from eduecon_radar.db import Database
from eduecon_radar.services import profile_store
from eduecon_radar.services.local_pipeline import run_profile_id
from eduecon_radar.services.migrations import overnight_schema_ready


@dataclass(frozen=True)
class OvernightExportPaths:
    mentor_csv: Path
    research_csv: Path
    digest_md: Path
    reading_queue_md: Path
    manual_abstract_queue_csv: Path
    provider_audit_csv: Path
    screening_audit_csv: Path


def _human_decision(review: dict | None) -> str:
    if not review:
        return "undecided"
    final = review.get("final_decision")
    if final == "include":
        return "related"
    if final == "exclude":
        return "irrelevant"
    if review.get("review_status") == "needs_revision":
        return "uncertain"
    return "undecided"


def collect_run_papers(
    database: Database,
    paper_ids: tuple[int, ...],
    *,
    task_id: int | None = None,
    sync_run_id: int | None = None,
    profile_id: str | None = None,
) -> list[dict[str, object]]:
    """Collect the facts and judgements for a set of papers.

    Judgement precedence is the same as everywhere else in the product: the
    profile assessment and profile review first, then the historical run-scoped
    columns for a database that predates migration 0013 or was never
    backfilled.  Reads never write, so this cannot drift.
    """
    normalized = tuple(dict.fromkeys(int(item) for item in paper_ids))
    if not normalized:
        return []
    placeholders = ",".join("?" for _ in normalized)
    resolved_profile = profile_id
    if resolved_profile is None and sync_run_id is not None:
        resolved_profile = run_profile_id(database, sync_run_id)
    assessments: dict[int, dict[str, object]] = {}
    profile_reviews: dict[int, dict[str, object]] = {}
    if resolved_profile and profile_store.profile_schema_ready(database):
        from eduecon_radar.services import profile_store as profile_store_module

        assessments = {
            paper_id: item.classification
            for paper_id, item in profile_store_module.assessments_for(
                database, resolved_profile, normalized
            ).items()
        }
        profile_reviews = {
            paper_id: {
                "human_decision": item.human_decision,
                "reviewer_notes": item.human_note,
            }
            for paper_id, item in profile_store_module.reviews_for(
                database, resolved_profile, normalized
            ).items()
        }
    sql = f"""SELECT p.id,p.doi,p.title,p.abstract,p.authors_json,p.publisher,
        p.online_date,p.issue_date,p.source_url,p.source_metadata_json,
        j.name journal,pt.chinese_title,pt.chinese_abstract,
        pt.abstract_translation_source,pt.title_translation_status,
        pt.abstract_translation_status
        FROM papers p JOIN journals j ON j.id=p.journal_id
        LEFT JOIN paper_translations pt ON pt.paper_id=p.id
        WHERE p.id IN ({placeholders}) ORDER BY p.id"""
    with database.connect() as connection:
        rows = [dict(row) for row in connection.execute(sql, list(normalized)).fetchall()]
        reviews: dict[int, dict] = {}
        if task_id is not None:
            # Historical path: the retired workflow stored reviews per task.
            for row in connection.execute(
                """SELECT paper_id,review_status,final_decision,reviewer_notes
                FROM workflow_task_reviews WHERE task_id=?""",
                (task_id,),
            ).fetchall():
                reviews[int(row["paper_id"])] = dict(row)
        else:
            for row in connection.execute(
                """SELECT paper_id,status review_status,reviewer_note
                FROM reviews WHERE paper_id IN ("""
                + placeholders
                + """)""",
                list(normalized),
            ).fetchall():
                item = dict(row)
                item["final_decision"] = (
                    "include" if item.get("review_status") == "approved" else
                    "exclude" if item.get("review_status") == "rejected" else None
                )
                reviews[int(item["paper_id"])] = item
        screening_rows: dict[int, dict] = {}
        status_rows: dict[int, dict] = {}
        if sync_run_id is not None and overnight_schema_ready(database):
            for row in connection.execute(
                """SELECT paper_id,human_decision,human_note,classification_json
                FROM paper_processing_status WHERE sync_run_id=?""",
                (sync_run_id,),
            ).fetchall():
                status_rows[int(row["paper_id"])] = dict(row)
        if overnight_schema_ready(database):
            for row in connection.execute(
                """SELECT ps.* FROM paper_screenings ps
                WHERE ps.paper_id IN ("""
                + placeholders
                + """) AND ps.stale=0
                ORDER BY ps.id DESC""",
                list(normalized),
            ).fetchall():
                paper_id = int(row["paper_id"])
                if paper_id not in screening_rows:
                    screening_rows[paper_id] = dict(row)
    papers = []
    for row in rows:
        paper_id = int(row["id"])
        metadata = json.loads(row.pop("source_metadata_json") or "{}")
        audit = metadata.get("abstract_enrichment")
        if not isinstance(audit, dict):
            audit = {}
        screening = screening_rows.get(paper_id) or {}
        status = status_rows.get(paper_id) or {}
        classification: dict[str, object] = {}
        try:
            parsed = json.loads(status.get("classification_json") or "{}")
        except (TypeError, ValueError):
            parsed = {}
        if isinstance(parsed, dict):
            classification = parsed
        # Canonical judgement first: the profile owns it.  The run-scoped
        # column is only a historical fallback.
        classification = assessments.get(paper_id) or classification
        review = profile_reviews.get(paper_id) or reviews.get(paper_id) or {}
        human_decision = (
            review.get("human_decision")
            or status.get("human_decision")
            or _human_decision(reviews.get(paper_id))
        )
        papers.append(
            {
                "paper_id": paper_id,
                "doi": row["doi"] or "",
                "title": row["title"] or "",
                "chinese_title": row["chinese_title"] or "",
                "abstract": row["abstract"] or "",
                "chinese_abstract": row["chinese_abstract"] or "",
                "abstract_translation_source": row["abstract_translation_source"] or "",
                "authors": json.loads(row["authors_json"] or "[]"),
                "journal": row["journal"] or "",
                "publisher": row["publisher"] or "",
                "year": str(row["issue_date"] or row["online_date"] or "")[:4],
                "publication_date": row["issue_date"] or row["online_date"] or "",
                "source_url": row["source_url"] or "",
                "abstract_source": metadata.get("abstract_source") or "",
                "provider_attempts": list(audit.get("attempts") or []),
                "abstract_final_reason": audit.get("final_reason") or "",
                "screening": screening,
                "classification": classification,
                "review": review,
                "human_decision": human_decision,
                "human_note": review.get("human_note")
                or review.get("reviewer_notes")
                or review.get("reviewer_note")
                or status.get("human_note")
                or "",
            }
        )
    return papers


def _screening_values(paper: dict[str, object]) -> dict[str, object]:
    screening = paper.get("screening") or {}
    classification = paper.get("classification")
    if not isinstance(screening, dict) or not screening:
        return {
            "automatic_decision": "",
            "judgment_kind": _judgment_kind({}, classification),
            "education_centrality": "",
            "economic_relevance": "",
            "evidence_sufficiency": "",
            "confidence": "",
            "evidence": [],
            "topic_tags": [],
            "model": "",
            "prompt_version": "",
            "reason": "",
        }
    return {
        "automatic_decision": screening.get("final_model_decision")
        or screening.get("model_decision")
        or "",
        "judgment_kind": _judgment_kind(screening, paper.get("classification")),
        "education_centrality": screening.get("education_centrality") or "",
        "economic_relevance": screening.get("economic_relevance") or "",
        "evidence_sufficiency": screening.get("evidence_sufficiency") or "",
        "confidence": screening.get("confidence") or "",
        "evidence": json.loads(screening.get("validated_evidence_json") or "[]")
        or json.loads(screening.get("evidence_json") or "[]"),
        "topic_tags": json.loads(screening.get("topic_tags_json") or "[]"),
        "model": screening.get("model_name") or "",
        "prompt_version": screening.get("prompt_version") or "",
        "reason": screening.get("reason") or "",
    }


def _judgment_kind(
    screening: dict[str, object], classification: dict[str, object] | None = None
) -> str:
    """How much evidence the judgment had, in the shared user-visible vocabulary.

    An export that lists a relevance decision without saying whether it came
    from a title-only screening or from the full abstract invites the reader to
    over-trust it.  Uses the same classifier as the UI so the wording cannot
    drift.

    ``paper_processing_status.classification_json`` is the authoritative record
    and is consulted first: ``paper_screenings.raw_response_json`` carries the
    basis only for the runs that wrote that key, so relying on it alone
    mislabelled 172 recovered papers as "basis not recorded".  When neither
    source has a basis, the explicit historical label is returned rather than
    claiming a complete judgment.
    """
    from eduecon_radar.services.paper_lifecycle_state import (
        screening_state_from_classification,
        screening_state_label,
    )

    decision = screening.get("final_model_decision") or screening.get("model_decision")
    basis: object = None

    if isinstance(classification, dict) and classification:
        decision = classification.get("decision") or decision
        basis = classification.get("screening_basis")

    if basis is None:
        raw = screening.get("raw_response_json")
        if raw:
            try:
                payload = json.loads(str(raw))
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                basis = payload.get("screening_basis")
    if basis is None:
        basis = screening.get("screening_basis")

    return screening_state_label(
        screening_state_from_classification(
            {"decision": decision, "screening_basis": basis}
        )
    )


def _provider_trail(paper: dict[str, object]) -> str:
    attempts = paper.get("provider_attempts") or []
    if not attempts:
        return ""
    compact = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        compact.append(
            {
                "provider": attempt.get("provider") or attempt.get("source"),
                "query_mode": attempt.get("query_mode"),
                "status": attempt.get("status") or attempt.get("result_state"),
                "http_status": attempt.get("http_status"),
                "retryable": attempt.get("retryable"),
                "error": attempt.get("error_category") or attempt.get("error_message"),
                "abstract": bool(attempt.get("abstract_text")),
                "fused_skip": bool(attempt.get("fused_skip")),
            }
        )
    return json.dumps(compact, ensure_ascii=False)


def _csv_text(rows: list[dict[str, object]], fields: list[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def _digest_markdown(papers: list[dict[str, object]]) -> str:
    lines = ["# 文献摘要清单 / Literature Digest", ""]
    for paper in papers:
        screening = _screening_values(paper)
        evidence = screening["evidence"]
        evidence_text = "\n".join(
            f"- {item.get('quote', '')} (supports: {item.get('supports', '')})"
            for item in evidence
            if isinstance(item, dict)
        )
        lines.extend(
            [
                f"# {paper['chinese_title'] or paper['title']}",
                f"{paper['title']}",
                "",
                f"- 作者: {'; '.join(paper['authors'])}",
                f"- 期刊: {paper['journal']}",
                f"- 年份: {paper['year']}",
                f"- DOI: {paper['doi']}",
                f"- 摘要来源: {paper['abstract_source'] or '无'}",
                "",
                "## 中文摘要",
                "",
                paper["chinese_abstract"] or "暂无中文摘要。",
                "",
                "## 英文摘要",
                "",
                paper["abstract"] or "暂无英文摘要。",
                "",
                "## 自动筛选",
                "",
                f"- 判定: {screening['automatic_decision'] or '未筛选'}",
                f"- 教育中心性: {screening['education_centrality']}",
                f"- 经济学实质性: {screening['economic_relevance']}",
                f"- 证据充分性: {screening['evidence_sufficiency']}",
                f"- 置信度: {screening['confidence']}",
                f"- 主题标签: {', '.join(screening['topic_tags'])}",
                f"- 理由: {screening['reason']}",
                "",
                "## 原文证据",
                "",
                evidence_text or "- 无",
                "",
                "## 人工决定",
                "",
                f"- {paper['human_decision']}",
                "",
                "## 用户备注",
                "",
                paper["human_note"] or "- 无",
                "",
            ]
        )
    return "\n".join(lines)


def _reading_queue(papers: list[dict[str, object]]) -> str:
    def sort_key(paper: dict[str, object]) -> tuple[int, str, int]:
        screening = _screening_values(paper)
        decision = str(paper["human_decision"])
        auto = str(screening["automatic_decision"])
        priority = (
            0
            if decision == "related"
            else 1
            if auto == "relevant"
            else 2
            if decision == "uncertain" or auto == "uncertain"
            else 3
            if not paper["abstract"]
            else 4
        )
        return (priority, str(paper["publication_date"] or ""), paper["paper_id"])

    ordered = sorted(papers, key=sort_key)
    groups = [
        ("立即精读（人工相关）", [p for p in ordered if p["human_decision"] == "related"]),
        (
            "值得浏览（自动相关且证据通过）",
            [
                p
                for p in ordered
                if p["human_decision"] == "undecided"
                and _screening_values(p)["automatic_decision"] == "relevant"
            ],
        ),
        (
            "待人工确认",
            [
                p
                for p in ordered
                if p["human_decision"] in {"undecided", "uncertain"}
                and _screening_values(p)["automatic_decision"] not in {"relevant", ""}
            ],
        ),
        ("待补摘要", [p for p in ordered if not p["abstract"]]),
        ("排除", [p for p in ordered if p["human_decision"] == "irrelevant"]),
    ]
    lines = ["# 阅读队列 / Reading Queue", ""]
    for title, items in groups:
        lines.append(f"## {title}")
        if not items:
            lines.append("- 无")
        for paper in items:
            screening = _screening_values(paper)
            lines.append(
                f"- [{paper['paper_id']}] {paper['chinese_title'] or paper['title']} "
                f"({paper['journal']} {paper['year']}) DOI: {paper['doi']} "
                f"自动: {screening['automatic_decision'] or '-'} "
                f"人工: {paper['human_decision']}"
            )
        lines.append("")
    return "\n".join(lines)


def write_overnight_exports(
    database: Database,
    paper_ids: tuple[int, ...],
    *,
    task_id: int | None = None,
    sync_run_id: int | None = None,
    export_dir: Path | None = None,
    profile_id: str | None = None,
) -> OvernightExportPaths:
    papers = collect_run_papers(
        database,
        paper_ids,
        task_id=task_id,
        sync_run_id=sync_run_id,
        profile_id=profile_id,
    )
    directory = export_dir or Path("exports") / "overnight_output"
    directory.mkdir(parents=True, exist_ok=True)

    mentor_rows = []
    research_rows = []
    manual_rows = []
    provider_rows = []
    screening_rows = []
    for paper in papers:
        screening = _screening_values(paper)
        mentor_rows.append(
            {
                "paper_id": paper["paper_id"],
                "中文标题": paper["chinese_title"],
                "英文标题": paper["title"],
                "作者": "; ".join(paper["authors"]),
                "期刊": paper["journal"],
                "年份": paper["year"],
                "DOI": paper["doi"],
                "中文摘要": paper["chinese_abstract"],
                "自动判断": screening["automatic_decision"],
                "判断依据": screening["judgment_kind"],
                "人工决定": paper["human_decision"],
                "判断理由": screening["reason"],
            }
        )
        research_rows.append(
            {
                **mentor_rows[-1],
                "英文摘要": paper["abstract"],
                "摘要来源": paper["abstract_source"],
                "Provider轨迹": _provider_trail(paper),
                "education_centrality": screening["education_centrality"],
                "economic_relevance": screening["economic_relevance"],
                "evidence_sufficiency": screening["evidence_sufficiency"],
                "confidence": screening["confidence"],
                "evidence": " | ".join(
                    item.get("quote", "")
                    for item in screening["evidence"]
                    if isinstance(item, dict)
                ),
                "topic_tags": ", ".join(screening["topic_tags"]),
                "model": screening["model"],
                "prompt_version": screening["prompt_version"],
                "用户备注": paper["human_note"],
            }
        )
        if not paper["abstract"]:
            manual_rows.append(
                {
                    "paper_id": paper["paper_id"],
                    "标题": paper["title"],
                    "中文标题": paper["chinese_title"],
                    "DOI": paper["doi"],
                    "出版方": paper["publisher"],
                    "尝试来源": "; ".join(
                        dict.fromkeys(
                            str(a.get("provider") or a.get("source"))
                            for a in paper["provider_attempts"]
                            if isinstance(a, dict)
                        )
                    ),
                    "最后失败原因": paper["abstract_final_reason"],
                    "DOI链接": (
                        f"https://doi.org/{paper['doi']}" if paper["doi"] else ""
                    ),
                }
            )
        for attempt in paper["provider_attempts"]:
            if not isinstance(attempt, dict):
                continue
            provider_rows.append(
                {
                    "paper_id": paper["paper_id"],
                    "provider": attempt.get("provider") or attempt.get("source") or "",
                    "query_mode": attempt.get("query_mode") or "",
                    "status": attempt.get("status") or attempt.get("result_state") or "",
                    "http_status": attempt.get("http_status") or "",
                    "retryable": attempt.get("retryable") or "",
                    "retry_after": attempt.get("retry_after") or "",
                    "next_retry_at": attempt.get("next_retry_at") or "",
                    "error_category": attempt.get("error_category") or "",
                    "error_message": attempt.get("error_message") or "",
                    "matched_doi": attempt.get("matched_doi") or "",
                    "matched_title": attempt.get("matched_title") or "",
                    "abstract": bool(attempt.get("abstract_text")),
                    "fused_skip": bool(attempt.get("fused_skip")),
                    "cache_hit": bool(attempt.get("cache_hit")),
                }
            )
        screening = paper.get("screening") or {}
        if screening:
            screening_rows.append(
                {
                    "paper_id": paper["paper_id"],
                    "input_hash": screening.get("input_hash") or "",
                    "model_decision": screening.get("model_decision") or "",
                    "final_model_decision": screening.get("final_model_decision") or "",
                    "education_centrality": screening.get("education_centrality") or "",
                    "economic_relevance": screening.get("economic_relevance") or "",
                    "evidence_sufficiency": screening.get("evidence_sufficiency") or "",
                    "confidence": screening.get("confidence") or "",
                    "validation_status": screening.get("validation_status") or "",
                    "validation_errors": screening.get("validation_errors_json") or "[]",
                    "review_decision": screening.get("review_decision") or "",
                    "disagreement_reason": screening.get("disagreement_reason") or "",
                    "evidence": (
                        screening.get("validated_evidence_json")
                        or screening.get("evidence_json")
                        or "[]"
                    ),
                    "reason": screening.get("reason") or "",
                    "model": screening.get("model_name") or "",
                    "prompt_version": screening.get("prompt_version") or "",
                    "screened_at": screening.get("screened_at") or "",
                    "stale": screening.get("stale") or 0,
                }
            )

    paths = OvernightExportPaths(
        mentor_csv=directory / "mentor_literature.csv",
        research_csv=directory / "research_literature.csv",
        digest_md=directory / "literature_digest.md",
        reading_queue_md=directory / "reading_queue.md",
        manual_abstract_queue_csv=directory / "manual_abstract_queue.csv",
        provider_audit_csv=directory / "provider_audit.csv",
        screening_audit_csv=directory / "screening_audit.csv",
    )
    paths.mentor_csv.write_text(
        _csv_text(
            mentor_rows,
            [
                "paper_id", "中文标题", "英文标题", "作者", "期刊", "年份", "DOI",
                "中文摘要", "自动判断", "判断依据", "人工决定", "判断理由",
            ],
        ),
        encoding="utf-8-sig",
    )
    paths.research_csv.write_text(
        _csv_text(
            research_rows,
            [
                "paper_id", "中文标题", "英文标题", "作者", "期刊", "年份", "DOI",
                "中文摘要", "自动判断", "判断依据", "人工决定", "判断理由",
                "英文摘要", "摘要来源",
                "Provider轨迹", "education_centrality", "economic_relevance",
                "evidence_sufficiency", "confidence", "evidence", "topic_tags",
                "model", "prompt_version", "用户备注",
            ],
        ),
        encoding="utf-8-sig",
    )
    paths.digest_md.write_text(_digest_markdown(papers), encoding="utf-8")
    paths.reading_queue_md.write_text(
        _reading_queue(papers), encoding="utf-8"
    )
    paths.manual_abstract_queue_csv.write_text(
        _csv_text(
            manual_rows,
            [
                "paper_id", "标题", "中文标题", "DOI", "出版方", "尝试来源",
                "最后失败原因", "DOI链接",
            ],
        ),
        encoding="utf-8-sig",
    )
    paths.provider_audit_csv.write_text(
        _csv_text(
            provider_rows,
            [
                "paper_id", "provider", "query_mode", "status", "http_status",
                "retryable", "retry_after", "next_retry_at", "error_category",
                "error_message", "matched_doi", "matched_title", "abstract",
                "fused_skip", "cache_hit",
            ],
        ),
        encoding="utf-8-sig",
    )
    paths.screening_audit_csv.write_text(
        _csv_text(
            screening_rows,
            [
                "paper_id", "input_hash", "model_decision", "final_model_decision",
                "education_centrality", "economic_relevance", "evidence_sufficiency",
                "confidence", "validation_status", "validation_errors",
                "review_decision", "disagreement_reason", "evidence", "reason",
                "model", "prompt_version", "screened_at", "stale",
            ],
        ),
        encoding="utf-8-sig",
    )
    return paths


def overnight_stats(papers: list[dict[str, object]]) -> dict[str, int]:
    stats = {
        "papers": len(papers),
        "with_abstract": 0,
        "chinese_title": 0,
        "chinese_abstract": 0,
        "screened": 0,
        "relevant": 0,
        "irrelevant": 0,
        "uncertain": 0,
        "needs_abstract": 0,
        "screening_failed": 0,
        "human_decided": 0,
        "provider_attempts": 0,
        "provider_success": 0,
        "rate_limited": 0,
        "timeout": 0,
        "fused_skip": 0,
    }
    for paper in papers:
        if paper["abstract"]:
            stats["with_abstract"] += 1
        if paper["chinese_title"]:
            stats["chinese_title"] += 1
        if paper["chinese_abstract"]:
            stats["chinese_abstract"] += 1
        if paper["human_decision"] != "undecided":
            stats["human_decided"] += 1
        screening = paper.get("screening") or {}
        if screening:
            stats["screened"] += 1
            final = str(screening.get("final_model_decision") or "")
            if final == "relevant":
                stats["relevant"] += 1
            elif final == "irrelevant":
                stats["irrelevant"] += 1
            elif final == "uncertain":
                stats["uncertain"] += 1
            elif str(screening.get("model_decision")) == "needs_abstract":
                stats["needs_abstract"] += 1
            if str(screening.get("validation_status")) in {
                "parse_failed",
                "recheck_failed",
            }:
                stats["screening_failed"] += 1
        for attempt in paper.get("provider_attempts") or []:
            if not isinstance(attempt, dict):
                continue
            stats["provider_attempts"] += 1
            status = str(attempt.get("status") or attempt.get("result_state") or "")
            if status == "success" or attempt.get("abstract_text"):
                stats["provider_success"] += 1
            if status == "rate_limited" or str(attempt.get("result_state")) in {
                "rate_limited",
                "provider_fused_skipped",
            }:
                stats["rate_limited"] += 1
            if status == "timeout":
                stats["timeout"] += 1
            if attempt.get("fused_skip"):
                stats["fused_skip"] += 1
    return stats
