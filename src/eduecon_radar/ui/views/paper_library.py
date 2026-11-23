"""论文库 — 可筛选、可导出的长期文献库。"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, replace
from datetime import date
from html import escape

import streamlit as st

from eduecon_radar.db import Database
from eduecon_radar.services import profile_store
from eduecon_radar.services.local_pipeline import (
    library_processing_run,
    reprocess_library_paper,
    rescreen_paper,
    save_human_decisions,
)
from eduecon_radar.services.paper_lifecycle import (
    preview_paper_removal,
    remove_paper,
)
from eduecon_radar.services.paper_lifecycle_state import (
    reprocessing_label,
    screening_state_from_classification,
    screening_state_label,
    translation_ready,
    usable_translation_sql,
)
from eduecon_radar.services.profile_registry import DEFAULT_PROFILE_ID
from eduecon_radar.services.reading_desk import ReadingDesk, identity, matching_topics, to_ris
from eduecon_radar.services.source_traceability import (
    build_doi_url,
    resolve_paper_access,
)
from eduecon_radar.services.ui_labels import (
    AUTO_DECISION_ZH,
    HUMAN_DECISION_OPTIONS,
    human_decision_from_zh,
    zh_abstract_source,
    zh_auto_decision,
    zh_human_decision,
    zh_status,
)
from eduecon_radar.ui.theme import (
    render_empty_state,
    render_section_header,
    status_badge_html,
)

PAGE_SIZE = 20
#: Export ranges, phrased as the question a researcher asks ("what goes in the
#: file?") rather than as an internal filter name.  The stored keys are also the
#: API used by the lifecycle tests, so they stay stable.
LIBRARY_EXPORT_SCOPES = (
    "当前筛选结果",
    "论文库全部",
    "人工相关",
    "人工不确定",
    "待审核",
)
LIBRARY_EXPORT_PRESETS = ("简洁版", "阅读版", "研究版")

#: The three artefacts the product actually produces, in the user's words.
#: Each entry is (scope, preset, purpose) and becomes one labelled download.
EXPORT_CHOICES: tuple[tuple[str, str, str, str], ...] = (
    (
        "阅读清单",
        "人工相关",
        "简洁版",
        "只看记录：标题、作者、期刊、年份、DOI。适合自己快速核对读了什么。",
    ),
    (
        "研究数据库",
        "论文库全部",
        "研究版",
        "完整字段：中英文摘要、自动判断、判断类型、人工决定与个人备注。适合长期归档与分析。",
    ),
    (
        "导师分享版",
        "人工相关",
        "阅读版",
        "只含有依据的内容：原始字段、正式翻译与摘要，不含 AI 判断和个人工作信息。",
    ),
)




def _has_processing_status(database: Database) -> bool:
    with database.connect() as connection:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='paper_processing_status'"
            ).fetchone()
            is not None
        )


@dataclass(frozen=True)
class LibraryScope:
    """Which profile's judgement the library shows.

    The profile tables are the authority.  Two historical fallbacks keep an
    un-migrated or not-yet-backfilled database readable, and both are scoped to
    the profile's *own* runs so one profile's legacy row can never leak into
    another profile's view:

    * a paper with no profile assessment still shows the run-scoped
      ``classification_json`` written before the single-authority cutover;
    * a paper with no profile review still shows the run-scoped human decision.

    Neither fallback is ever written by the product path.
    """

    profile_id: str | None = None
    profile_ready: bool = False
    shared: bool = False

    @property
    def uses_profile(self) -> bool:
        return self.profile_ready and bool(self.profile_id)

    @property
    def classification_column(self) -> str:
        if not self.uses_profile:
            return "ps.classification_json"
        # Canonical first, historical run snapshot second.
        return "COALESCE(pa.classification_json, ps.classification_json)"

    @property
    def human_decision_column(self) -> str:
        if not self.uses_profile:
            return "ps.human_decision"
        return "COALESCE(pr.human_decision, ps.human_decision)"

    @property
    def human_note_column(self) -> str:
        if not self.uses_profile:
            return "ps.human_note"
        return "CASE WHEN pr.paper_id IS NULL THEN ps.human_note ELSE pr.human_note END"

    @property
    def model_column(self) -> str:
        if not self.uses_profile:
            return "ps.model_name"
        return "COALESCE(pa.model_name, ps.model_name)"

    @property
    def prompt_column(self) -> str:
        if not self.uses_profile:
            return "ps.prompt_version"
        return "COALESCE(pa.prompt_version, ps.prompt_version)"

    def selected_columns(self) -> str:
        """The judgement columns, built from the same expressions as the filters.

        Selecting different expressions from the ones the predicates filter on
        is how a row can be found by a filter and then displayed as undecided;
        both now come from this one definition.
        """
        fingerprint = "pa.input_fingerprint" if self.uses_profile else "NULL"
        return (
            f"{fingerprint} input_fingerprint,"
            f"ps.status processing_status,{self.classification_column} classification_json,"
            f"{self.human_decision_column} human_decision,"
            f"{self.human_note_column} human_note,"
            f"{self.model_column} model_name,{self.prompt_column} prompt_version"
        )

    def join_values(self) -> list[object]:
        if not self.uses_profile:
            return []
        # Order follows the SQL text of _processing_status_sql: the profile's
        # processing rows, then the assessment join, then the review join.
        return [self.profile_id, self.profile_id, self.profile_id]

    def membership_clause(self) -> tuple[str, list[object]]:
        if not self.uses_profile:
            return "", []
        return (
            f"p.id IN ({profile_store.MEMBERSHIP_SQL})",
            [self.profile_id] * profile_store.MEMBERSHIP_PARAMS,
        )


def resolve_library_scope(database: Database, profile_id: str | None) -> LibraryScope:
    ready = profile_store.profile_schema_ready(database)
    if profile_id is None:
        return LibraryScope(profile_id=None, profile_ready=ready)
    return LibraryScope(profile_id=profile_id, profile_ready=ready)


def default_library_scope(database: Database) -> LibraryScope:
    """The scope used when a caller does not name a profile.

    It resolves to the default profile rather than to the legacy run column, so
    a forgotten argument can never silently read a stale or empty judgement.
    """
    return resolve_library_scope(database, DEFAULT_PROFILE_ID)


def _library_predicates(
    *,
    journal: str,
    abstract_state: str,
    decision: str,
    auto_decision: str,
    year: str,
    search: str,
    has_status: bool,
    scope: LibraryScope | None = None,
) -> tuple[list[str], list[object]]:
    active = scope or LibraryScope()
    conditions = ["1=1"]
    values: list[object] = []
    if active.uses_profile and not active.shared:
        clause, clause_values = active.membership_clause()
        conditions.append(clause)
        values.extend(clause_values)
    if journal != "全部":
        conditions.append("j.name=?")
        values.append(journal)
    if abstract_state == "有摘要":
        conditions.append("p.abstract IS NOT NULL AND length(trim(p.abstract))>0")
    elif abstract_state == "缺摘要":
        conditions.append("(p.abstract IS NULL OR length(trim(p.abstract))=0)")
    if decision != "全部" and has_status:
        conditions.append(f"COALESCE({active.human_decision_column},'undecided')=?")
        values.append(human_decision_from_zh(decision))
    if auto_decision != "全部" and has_status:
        conditions.append(
            f"json_extract({active.classification_column}, '$.decision')=?"
        )
        values.append(
            next(
                (
                    key
                    for key, label in AUTO_DECISION_ZH.items()
                    if label == auto_decision
                ),
                None,
            )
        )
    if year != "全部":
        conditions.append("substr(COALESCE(p.issue_date,p.online_date),1,4)=?")
        values.append(year)
    if search:
        like = f"%{search.strip()}%"
        title_sql = usable_translation_sql("title")
        conditions.append(f"(p.title LIKE ? OR p.doi LIKE ? OR {title_sql} LIKE ?)")
        values.extend([like, like, like])
    return conditions, values


def _processing_status_sql(
    has_status: bool, scope: LibraryScope | None = None
) -> tuple[str, str]:
    active = scope or LibraryScope()
    if not has_status:
        return (
            "",
            """NULL processing_status,NULL classification_json,NULL human_decision,
            NULL human_note,NULL model_name,NULL prompt_version""",
        )
    run_state = """LEFT JOIN (
            SELECT a.* FROM paper_processing_status a
            JOIN (
                SELECT paper_id, MAX(sync_run_id) max_run
                FROM paper_processing_status GROUP BY paper_id
            ) b ON a.paper_id=b.paper_id AND a.sync_run_id=b.max_run
        ) ps ON ps.paper_id=p.id"""
    if active.uses_profile:
        # The run state shown belongs to this profile's latest run for the
        # paper, so a legacy run-scoped column can only ever be read back by the
        # profile that produced it.
        return (
            """LEFT JOIN (
            SELECT a.* FROM paper_processing_status a
            JOIN (
                SELECT ps2.paper_id, MAX(ps2.sync_run_id) max_run
                FROM paper_processing_status ps2
                JOIN sync_runs sr2 ON sr2.id=ps2.sync_run_id
                WHERE sr2.profile_id=?
                GROUP BY ps2.paper_id
            ) b ON a.paper_id=b.paper_id AND a.sync_run_id=b.max_run
        ) ps ON ps.paper_id=p.id
        LEFT JOIN paper_profile_assessments pa
               ON pa.paper_id=p.id AND pa.profile_id=?
        LEFT JOIN paper_profile_reviews pr
               ON pr.paper_id=p.id AND pr.profile_id=?""",
            active.selected_columns(),
        )
    return (run_state, active.selected_columns())


def _library_papers(
    database: Database,
    *,
    journal: str,
    abstract_state: str,
    decision: str,
    auto_decision: str,
    year: str,
    search: str,
    limit: int | None = None,
    offset: int = 0,
    paper_ids: tuple[int, ...] | None = None,
    summary_only: bool = False,
    has_status: bool | None = None,
    scope: LibraryScope | None = None,
) -> list[dict[str, object]]:
    if has_status is None:
        has_status = _has_processing_status(database)
    active = scope if scope is not None else default_library_scope(database)
    conditions, values = _library_predicates(
        journal=journal,
        abstract_state=abstract_state,
        decision=decision,
        auto_decision=auto_decision,
        year=year,
        search=search,
        has_status=has_status,
        scope=active,
    )
    ps_join, ps_columns = _processing_status_sql(has_status, active)
    if paper_ids is not None:
        conditions.append("p.id IN (SELECT value FROM json_each(?))")
        values.append(json.dumps(paper_ids))
    detail = ("NULL AS authors_json,NULL AS source_metadata_json" if summary_only
              else "p.authors_json,p.source_metadata_json")
    pagination = ""
    if limit is not None:
        pagination = "LIMIT ? OFFSET ?"
        values.extend((limit, max(0, offset)))
    sql = f"""
        SELECT p.id,p.title,p.abstract,{detail},p.doi,p.source_url,
               p.open_access_url,p.volume,p.issue,p.publisher,
               COALESCE(p.issue_date,p.online_date) publication_date,
               j.name journal,{usable_translation_sql("title")} chinese_title,
               {usable_translation_sql("abstract")} chinese_abstract,pt.abstract_translation_status,
               pt.abstract_translation_source,{ps_columns}
        FROM papers p
        JOIN journals j ON j.id=p.journal_id
        LEFT JOIN paper_translations pt ON pt.paper_id=p.id
        {ps_join}
        WHERE {" AND ".join(conditions)}
        ORDER BY p.id DESC {pagination}
    """
    with database.connect() as connection:
        rows = connection.execute(sql, [*active.join_values(), *values]).fetchall()
    papers = []
    for row in rows:
        item = dict(row)
        metadata = json.loads(item.pop("source_metadata_json") or "{}")
        item["source_metadata"] = metadata
        item["authors"] = json.loads(item.pop("authors_json") or "[]")
        item["abstract_source"] = metadata.get("abstract_source")
        item["profile_id"] = active.profile_id
        item["processing"] = {
            "status": item.pop("processing_status", None) or "pending",
            "classification": json.loads(item.pop("classification_json", None) or "{}"),
            "model_name": item.pop("model_name", None),
            "prompt_version": item.pop("prompt_version", None),
        }
        profile_store.reflect_saved_assessment(item, item.pop("input_fingerprint", None))
        item["human_decision"] = item.pop("human_decision", None) or "undecided"
        item["reviewer_notes"] = item.pop("human_note", None)
        papers.append(item)
    return papers


def radar_page(database, scope, section, topics, size, today):
    """Rank only unread records, loading full paper details just for visible cards."""
    join, _ = _processing_status_sql(True, scope)
    membership, members = scope.membership_clause()
    discovered = ""
    added = "NULL"
    added_join = ""
    values = []
    if section == "本周新收":
        discovered = """WITH appearances AS (
            SELECT CAST(j.value AS INTEGER) paper, r.started_at stamp
            FROM sync_runs r, json_each(r.matched_paper_ids) j WHERE r.profile_id=?
              AND r.requested_limit IS NOT NULL AND json_valid(r.matched_paper_ids)
            UNION ALL
            SELECT ps.paper_id, r.started_at FROM paper_processing_status ps
            JOIN sync_runs r ON r.id=ps.sync_run_id WHERE r.profile_id=?
              AND r.requested_limit IS NOT NULL
        ), first_seen AS (SELECT paper,date(min(stamp),'localtime') added
                         FROM appearances GROUP BY paper),"""
        values.extend((scope.profile_id, scope.profile_id))
        added = "fs.added"
        added_join = "JOIN first_seen fs ON fs.paper=p.id"
    hits = (f"topic_hits(p.title,p.abstract,{usable_translation_sql('title')},"
            f"{usable_translation_sql('abstract')})" if topics else "0")
    order = "hits DESC, (ai='relevant') DESC, (ai!='irrelevant') DESC, published DESC,id DESC"
    condition = ""
    if section in ("近期发表", "本周新收"):
        column, days = ("published", 29) if section == "近期发表" else ("added", 6)
        order = f"{column} DESC,id DESC"
        condition = f"WHERE {column} BETWEEN date(?,'-{days} day') AND ?"
    sql = f"""{discovered or 'WITH'} candidates AS MATERIALIZED (
       SELECT p.id, {added} added, {hits} hits,
              CASE WHEN p.issue_date IS NOT NULL AND p.online_date IS NOT NULL
                   THEN min(p.issue_date,p.online_date)
                   ELSE COALESCE(p.issue_date,p.online_date) END published,
              COALESCE(json_extract({scope.classification_column},'$.decision'),'') ai
       FROM papers p LEFT JOIN paper_translations pt ON pt.paper_id=p.id
       {join} {added_join}
       WHERE {membership} AND COALESCE({scope.human_decision_column},'undecided')='undecided'
    ) SELECT id,added,published,count(*) OVER () total FROM candidates {condition}
    ORDER BY {order} LIMIT ?"""
    values.extend([*scope.join_values(), *members])
    if condition:
        values.extend([today.isoformat(), today.isoformat()])
    with database.connect() as conn:
        if topics:
            def count_hits(title, abstract, chinese_title, chinese_abstract):
                paper = dict(title=title, abstract=abstract, chinese_title=chinese_title,
                             chinese_abstract=chinese_abstract)
                return len(matching_topics(paper, topics))
            conn.create_function("topic_hits", 4, count_hits, deterministic=True)
        rows = conn.execute(sql, [*values, size]).fetchall()
    if not rows:
        return [], 0
    visible = _library_papers(database, journal="全部", abstract_state="全部", decision="全部",
                              auto_decision="全部", year="全部", search="", scope=scope,
                              paper_ids=tuple(row["id"] for row in rows), summary_only=True)
    by_id = {p["id"]: p for p in visible}
    selected = []
    for row in rows:
        paper = by_id.get(row["id"])
        if paper and paper["human_decision"] == "undecided":
            paper["added"] = row["added"]
            paper["publication_date"] = row["published"]
            selected.append((paper, matching_topics(paper, topics)))
    return selected, rows[0]["total"]


def _library_count(
    database: Database,
    *,
    journal: str,
    abstract_state: str,
    decision: str,
    auto_decision: str,
    year: str,
    search: str,
    has_status: bool,
    scope: LibraryScope | None = None,
) -> int:
    active = scope if scope is not None else default_library_scope(database)
    conditions, values = _library_predicates(
        journal=journal,
        abstract_state=abstract_state,
        decision=decision,
        auto_decision=auto_decision,
        year=year,
        search=search,
        has_status=has_status,
        scope=active,
    )
    ps_join, _ = _processing_status_sql(has_status, active)
    sql = f"""SELECT COUNT(*)
        FROM papers p
        JOIN journals j ON j.id=p.journal_id
        LEFT JOIN paper_translations pt ON pt.paper_id=p.id
        {ps_join}
        WHERE {" AND ".join(conditions)}"""
    with database.connect() as connection:
        return int(
            connection.execute(sql, [*active.join_values(), *values]).fetchone()[0]
        )


def _decision_tone(decision: object) -> str:
    return {
        "relevant": "complete",
        "irrelevant": "info",
        "uncertain": "review",
        "needs_abstract": "missing",
        "insufficient_evidence": "missing",
        "needs_retry": "retry",
        "processing": "pending",
    }.get(str(decision or ""), "pending")


def _human_tone(decision: object) -> str:
    return {
        "related": "complete",
        "irrelevant": "info",
        "uncertain": "review",
        "undecided": "pending",
    }.get(str(decision or "undecided"), "pending")


def _authors_text(authors: object) -> str:
    if not isinstance(authors, list):
        return ""
    names: list[str] = []
    for author in authors:
        if isinstance(author, str):
            names.append(author)
        elif isinstance(author, dict):
            name = " ".join(
                str(author.get(part) or "").strip()
                for part in ("given", "family")
            ).strip()
            if name:
                names.append(name)
    return ", ".join(names)


def _render_removal_controls(database: Database, paper: dict[str, object]) -> None:
    confirm_key = f"library_remove_confirm_{paper['id']}"
    if not st.session_state.get(confirm_key):
        if st.button("移除论文", key=f"library_remove_{paper['id']}"):
            st.session_state[confirm_key] = True
            st.rerun()
        return
    try:
        preview = preview_paper_removal(database, int(paper["id"]))
    except ValueError:
        st.session_state.pop(confirm_key, None)
        st.rerun()
        return
    warning = "论文将从论文库、审核和导出中移除，并保留可恢复的归档记录。"
    if preview.has_human_data:
        warning = (
            "这篇论文含人工编辑或决定。移除后不会再次同步进入论文库，"
            "人工数据会保留在归档记录中。"
        )
    st.warning(warning)
    confirmed = st.checkbox(
        "确认移除这篇论文",
        key=f"library_remove_checked_{paper['id']}",
    )
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button(
        "确认移除",
        disabled=not confirmed,
        key=f"library_remove_commit_{paper['id']}",
    ):
        remove_paper(database, int(paper["id"]))
        st.session_state.pop("library_export_data", None)
        st.session_state.pop(confirm_key, None)
        st.toast("论文已归档并移除")
        st.rerun()
    if cancel_col.button("取消", key=f"library_remove_cancel_{paper['id']}"):
        st.session_state.pop(confirm_key, None)
        st.rerun()


def _render_reprocess_action(
    database: Database,
    settings: object,
    paper: dict[str, object],
    profile: object | None = None,
) -> None:
    from eduecon_radar.services.pipeline_builder import (
        build_enrichment_service,
        build_local_pipeline_services,
    )

    notice_key = f"library_processing_notice_{paper['id']}"
    if message := st.session_state.pop(notice_key, None):
        st.info(message)
    processing = paper.get("processing") or {}
    classification = processing.get("classification") or {}
    has_en = bool(str(paper.get("abstract") or "").strip())
    has_cn = translation_ready(
        paper.get("chinese_abstract"), paper.get("abstract_translation_status"),
        paper.get("abstract_translation_source"),
    )
    label = reprocessing_label(
        has_en, has_cn, classification.get("decision"), classification.get("screening_basis")
    )
    if label is None:
        if has_en and getattr(settings, "deepseek_api_key", None) and st.button(
            "重新评估相关性", key=f"library_rescreen_{paper['id']}",
            help="使用当前方向的提示词重新初筛，保留你的决定和备注。",
        ):
            paper_id = int(paper["id"])
            try:
                with st.spinner("正在重新评估相关性…"):
                    services = build_local_pipeline_services(database, settings, profile=profile)
                    run_id = library_processing_run(database, paper_id, services.profile_id)
                    outcome = rescreen_paper(database, run_id, paper_id, services)
            except Exception:
                st.error("本次重新评估失败，可稍后重试。")
            else:
                if outcome.get("decision") in {"relevant", "irrelevant", "uncertain"}:
                    st.session_state.pop("library_export_data", None)
                    st.toast("已更新本方向的 AI 建议")
                    st.rerun()
                st.info("本次未获得可用的 AI 判断，可稍后重试。")
        return
    if not getattr(settings, "deepseek_api_key", None):
        if not has_en and st.button("查找公开摘要", key=f"library_enrich_{paper['id']}"):
            try:
                with st.spinner("正在查找公开摘要…"):
                    stats = build_enrichment_service(database, settings).enrich_papers(
                        (int(paper["id"]),), workflow_batch_id=None, force_refresh=True,
                    )
            except Exception:
                st.warning("摘要来源暂时无法访问，可稍后重试。")
            else:
                st.session_state[notice_key] = (
                    "已获得原始摘要" if stats.enriched or stats.skipped_have_abstract
                    else "本次未获得公开摘要，已保留检索记录"
                )
                st.session_state.pop("library_export_data", None)
                st.rerun()
        return
    if st.button(
        label,
        key=f"library_reprocess_{paper['id']}",
        use_container_width=True,
    ):
        try:
            services = build_local_pipeline_services(database, settings, profile=profile)
            with st.status(f"正在{label}…", expanded=True) as progress:
                progress.update(label="正在获取/补全英文摘要…")
                result = reprocess_library_paper(
                    database,
                    int(paper["id"]),
                    services,
                    force_abstract_refresh=True,
                )
                status = dict(result["status"] or {})
                state_label = {
                    "classified": "已更新本方向的自动判断",
                    "needs_retry": "部分步骤需要重试",
                    "needs_manual_abstract": "摘要源暂未提供，可稍后重试",
                }.get(str(status.get("status") or ""), "处理完成")
                progress.update(label=state_label, state="complete")
        except Exception as exc:
            st.error(f"{label}失败：{str(exc).replace('Authorization', '[redacted]')[:400]}")
        else:
            st.session_state[notice_key] = state_label
            st.session_state.pop("library_export_data", None)
            st.rerun()


_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_title(value: object) -> str:
    """Publisher titles carry JATS markup and hard-wrapped whitespace.

    Crossref returns titles such as ``Comment on\\n <i>‘…’</i>``.  Rendered
    verbatim they show a newline in the middle of a card and literal ``<i>``
    once escaped, which reads as a data defect rather than a citation.  The
    stored value is never modified; this only cleans what is displayed.
    """
    text = str(value or "")
    if not text:
        return ""
    text = _TAG_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _render_paper_card(
    database: Database,
    settings: object,
    paper: dict[str, object],
    profile: object | None = None,
    *, opened: bool = False,
) -> None:
    hosted = getattr(settings, "hosted", False)
    excerpt = paper.get("abstract_source") == "publisher_excerpt"
    processing = paper.get("processing") or {}
    classification = processing.get("classification") or {}
    auto_decision = classification.get("decision")
    human_decision = paper.get("human_decision") or "undecided"
    publication_year = str(paper.get("publication_date") or "")[:4] or "日期未知"
    chinese_title = _clean_title(paper.get("chinese_title"))
    english_title = _clean_title(paper.get("title")) or "Untitled"
    doi = str(paper.get("doi") or "").strip()
    doi_url = build_doi_url(doi)
    doi_html = (
        '<a class="edr-doi" target="_blank" rel="noopener noreferrer" '
        f'href="{escape(doi_url, quote=True)}">{escape(doi)}</a>'
        if doi_url
        else '<span class="edr-doi">DOI 不可用</span>'
    )
    title_html = (
        f'<div class="edr-library-title">{escape(chinese_title)}</div>'
        if chinese_title
        else ""
    )
    abstract_label = "期刊摘要节选" if excerpt else "有原始摘要"
    badges = status_badge_html(
        abstract_label if paper.get("abstract") else "暂无摘要",
        "complete" if paper.get("abstract") else "missing",
    )
    if auto_decision or not hosted:
        badges += status_badge_html(
            f"自动：{zh_auto_decision(auto_decision)}", _decision_tone(auto_decision),
        )
    badges += status_badge_html(
        f"人工：{zh_human_decision(human_decision)}", _human_tone(human_decision),
    )
    if not hosted:
        badges += status_badge_html(
            zh_status(processing.get("status")),
            "complete" if processing.get("status") == "classified" else "pending",
        )
    st.markdown(
        "".join(
            (
                '<div class="edr-library-card">',
                title_html,
                f'<div class="edr-library-title-en">{escape(english_title)}</div>',
                '<div class="edr-library-meta">',
                escape(str(paper.get("journal") or "期刊未知")),
                " · ",
                escape(publication_year),
                " · ",
                doi_html,
                "</div>",
                f'<div class="edr-badges">{badges}</div>',
                "</div>",
            )
        ),
        unsafe_allow_html=True,
    )
    details_key = f"library_details_{paper['id']}"
    profile_id = getattr(profile, "profile_id", None)
    def remember_open():
        if st.session_state.get(details_key) and profile_id:
            ReadingDesk(database).put("resume/" + profile_id,
                                      {"paper": paper["id"], "identity": identity(paper)})

    if opened or st.toggle(
        "查看摘要与详情",
        key=details_key, on_change=remember_open,
    ):
        authors = _authors_text(paper.get("authors"))
        if authors:
            st.caption(f"作者：{authors}")
        if paper.get("chinese_abstract"):
            st.markdown("**摘要节选 · 中文译文**" if excerpt else "**中文摘要**")
            st.markdown(
                '<div class="edr-reading-text">'
                f'{escape(str(paper["chinese_abstract"]))}</div>',
                unsafe_allow_html=True,
            )
        elif paper.get("abstract"):
            st.markdown("**原始摘要**")
            st.markdown(
                f'<div class="edr-reading-text">{escape(str(paper["abstract"]))}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"来源：{zh_abstract_source(paper.get('abstract_source'))}")
        else:
            st.caption("暂无摘要")
        if paper.get("chinese_abstract") and paper.get("abstract"):
            with st.expander("查看英文节选" if excerpt else "查看英文原文"):
                st.markdown(
                    f'<div class="edr-reading-text edr-reading-text--source">'
                    f'{escape(str(paper["abstract"]))}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"来源：{zh_abstract_source(paper.get('abstract_source'))}"
                )
        if reason := classification.get("reason"):
            with st.expander("本方向 AI 初筛理由"):
                st.write(reason)
        profile_id = getattr(profile, "profile_id", None)
        if profile_id and profile_store.profile_schema_ready(database):
            desk = ReadingDesk(database)
            saved = desk.shelf(profile_id).get(paper["id"]) == identity(paper)
            if st.button("移出待读书架" if saved else "加入待读书架", key=f"shelve_{paper['id']}"):
                desk.bookmark(profile_id, paper, not saved)
                st.rerun()
            notes = paper.get("source_metadata", {}).get("reading_notes", {})
            if reading_note := notes.get(profile_id):
                st.caption("阅读线索 · 编者整理")
                st.write(reading_note)
            with st.form(f"library_review_{profile_id}_{paper['id']}"):
                options = list(HUMAN_DECISION_OPTIONS)
                decision = st.radio(
                    "人工决定", options, index=options.index(zh_human_decision(human_decision)),
                    horizontal=True,
                )
                note = st.text_input("个人备注", value=str(paper.get("reviewer_notes") or ""),
                                     max_chars=1000 if hosted else None)
                if st.form_submit_button("保存人工决定"):
                    paper_id = int(paper["id"])
                    run_id = library_processing_run(database, paper_id, profile_id)
                    save_human_decisions(
                        database, run_id, {paper_id: (human_decision_from_zh(decision), note)},
                    )
                    st.session_state.pop("library_export_data", None)
                    st.toast("已保存人工决定")
                    st.rerun()
        elif paper.get("reviewer_notes"):
            st.write(paper["reviewer_notes"])
        if settings is not None and not hosted:
            _render_reprocess_action(database, settings, paper, profile)
        access = resolve_paper_access(
            doi=doi,
            open_access_url=str(paper.get("open_access_url") or "") or None,
            source_url=str(paper.get("source_url") or "") or None,
            source_metadata=dict(paper.get("source_metadata") or {}),
        )
        if access.preferred_url:
            st.markdown(
                '<a class="edr-access-link" target="_blank" '
                'rel="noopener noreferrer" '
                f'href="{escape(access.preferred_url, quote=True)}">'
                f'打开公开页面 · {escape(access.preferred_label or "外部来源")} ↗</a>',
                unsafe_allow_html=True,
            )
        if not hosted:
            st.divider()
            _render_removal_controls(database, paper)


def _export_scope_papers(
    database: Database,
    scope: str,
    *,
    filters: dict[str, str],
    has_status: bool,
    library_scope: LibraryScope | None = None,
) -> list[dict[str, object]]:
    if library_scope is None:
        library_scope = default_library_scope(database)
    defaults = {
        "journal": "全部",
        "abstract_state": "全部",
        "decision": "全部",
        "auto_decision": "全部",
        "year": "全部",
        "search": "",
    }
    if scope == "当前筛选结果":
        selected = {**defaults, **filters}
        return _library_papers(database, **selected, has_status=has_status, scope=library_scope)
    if scope == "人工相关":
        selected = {**defaults, "decision": "相关"}
        return _library_papers(database, **selected, has_status=has_status, scope=library_scope)
    if scope == "人工不确定":
        selected = {**defaults, "decision": "不确定"}
        return _library_papers(database, **selected, has_status=has_status, scope=library_scope)
    all_papers = _library_papers(
        database, **defaults, has_status=has_status, scope=library_scope
    )
    if scope == "待审核":
        return [
            paper
            for paper in all_papers
            if paper.get("human_decision") == "undecided"
        ]
    return all_papers


def _preset_rows(
    papers: list[dict[str, object]], preset: str
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for paper in papers:
        classification = dict(
            dict(paper.get("processing") or {}).get("classification") or {}
        )
        row = {
            "中文标题": str(paper.get("chinese_title") or ""),
            "英文标题": str(paper.get("title") or ""),
            "作者": _authors_text(paper.get("authors")),
            "期刊": str(paper.get("journal") or ""),
            "年份": str(paper.get("publication_date") or "")[:4],
            "DOI": str(paper.get("doi") or ""),
        }
        if preset in {"阅读版", "研究版"}:
            row.update(
                {
                    "英文摘要": str(paper.get("abstract") or ""),
                    "中文摘要": str(paper.get("chinese_abstract") or ""),
                    "摘要来源": zh_abstract_source(paper.get("abstract_source")),
                }
            )
        if preset == "研究版":
            row.update(
                {
                    "自动判断": zh_auto_decision(classification.get("decision")),
                    "判断类型": screening_state_label(
                        screening_state_from_classification(classification)
                    ),
                    "人工判断": zh_human_decision(paper.get("human_decision")),
                    "备注": str(paper.get("reviewer_notes") or ""),
                }
            )
        rows.append(row)
    return rows


def _rows_to_csv(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return "\ufeff" + output.getvalue()


def _rows_to_markdown(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        cells = [
            str(row.get(field) or "")
            .replace("|", "\\|")
            .replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
            for field in fields
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(
    database: Database, settings: object = None, *, profile: object | None = None
) -> None:
    profile_id = getattr(profile, "profile_id", None)
    library_scope = resolve_library_scope(database, profile_id)
    # The library is one store read through one profile's judgement.  The
    # heading names the screen; the line under it names whose judgement is on
    # screen, so a reader is never unsure which queue a badge belongs to.
    render_section_header("论文库")
    if profile_id:
        profile_label = getattr(profile, "display_name", None) or profile_id
        st.markdown(
            f'<div class="edr-library-scope">研究方向「{escape(str(profile_label))}」'
            " · 自动判断与人工决定按研究方向分别保存；论文元数据与摘要共用一份</div>",
            unsafe_allow_html=True,
        )
    has_status = _has_processing_status(database)
    with database.connect() as connection:
        journals = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT name FROM journals ORDER BY name"
            ).fetchall()
        ]
        years = [
            str(row[0])
            for row in connection.execute(
                """SELECT DISTINCT substr(COALESCE(issue_date, online_date),1,4)
                FROM papers WHERE issue_date IS NOT NULL OR online_date IS NOT NULL
                ORDER BY 1 DESC"""
            ).fetchall()
        ]
    with st.container(key="edr_library_filters"):
        if library_scope.uses_profile and st.segmented_control(
            "查找范围", ("本方向", "全部方向"), default="本方向", key="library_scope_picker",
            help="共用题名、摘要和译文。在这里保存的决定与备注归入当前研究方向。",
        ) == "全部方向":
            library_scope = replace(library_scope, shared=True)
        search = st.text_input(
            "搜索",
            key="library_search",
            placeholder="输入中文或英文标题、DOI",
            max_chars=200 if getattr(settings, "hosted", False) else None,
            help="在中文标题、英文标题和 DOI 中查找。",
        )
        # Three filters are used every session (whose judgement, which journal,
        # which year).  The rest describe the pipeline rather than the paper, so
        # they live one disclosure down instead of in the main filter row.
        primary = st.columns([1.35, 1.35, 0.9])
        quick_decision = primary[0].segmented_control(
            "人工决定",
            ("全部", "相关", "不确定", "无关", "未决定"),
            default="全部",
            key="library_quick_decision",
            help="只看某一类人工决定；期刊、年份与摘要条件可以叠加使用。",
        ) or "全部"
        if quick_decision in HUMAN_DECISION_OPTIONS:
            st.session_state["library_decision"] = quick_decision
            decision = quick_decision
        else:
            st.session_state["library_decision"] = "全部"
            decision = "全部"
        journal = primary[1].selectbox(
            "期刊", ["全部", *journals], key="library_journal"
        )
        year = primary[2].selectbox("年份", ["全部", *years], key="library_year")
        with st.expander("更多筛选（摘要与自动判断）"):
            advanced = st.columns(2)
            abstract_state = advanced[0].selectbox(
                "摘要状态", ["全部", "有摘要", "缺摘要"], key="library_abstract_state"
            )
            auto_decision = advanced[1].selectbox(
                "自动判断",
                ["全部", *AUTO_DECISION_ZH.values()],
                key="library_auto_decision",
            )
    selection = (library_scope, search, journal, year, decision, abstract_state, auto_decision)
    if st.session_state.get("library_filters") != selection:
        st.session_state["library_filters"] = selection
        st.session_state["library_page"] = 1
        st.session_state.pop("library_export_data", None)
    total = _library_count(
        database,
        journal=journal,
        abstract_state=abstract_state,
        decision=decision,
        auto_decision=auto_decision,
        year=year,
        search=search,
        has_status=has_status,
        scope=library_scope,
    )
    if total == 0:
        render_empty_state("当前筛选条件下没有论文。可以调整期刊、年份或判断状态。")
        return

    page_count = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    if int(st.session_state.get("library_page", 1)) > page_count:
        st.session_state["library_page"] = 1
    page = int(
        st.number_input(
            "页码",
            min_value=1,
            max_value=page_count,
            step=1,
            key="library_page",
        )
    )
    start = (page - 1) * PAGE_SIZE
    page_papers = _library_papers(
        database,
        journal=journal,
        abstract_state=abstract_state,
        decision=decision,
        auto_decision=auto_decision,
        year=year,
        search=search,
        limit=PAGE_SIZE,
        offset=start,
        has_status=has_status,
        scope=library_scope,
    )
    st.markdown(
        f'<div class="edr-library-count">共 {total} 篇 · '
        f"第 {page}/{page_count} 页 · 当前显示 {len(page_papers)} 篇</div>",
        unsafe_allow_html=True,
    )
    # Three artefacts under the names a researcher uses, each stating what ends
    # up in the file.  The scope/preset selectors still exist for the occasional
    # custom cut, but they are the exception rather than the entry point.
    with st.expander("导出", expanded=False):
        st.markdown(
            '<p class="edr-export-intro">选择要生成的文件。'
            "可下载 CSV 表格、Markdown 文本或 RIS 书目。</p>",
            unsafe_allow_html=True,
        )
        for index, (label, choice_scope, choice_preset, purpose) in enumerate(
            EXPORT_CHOICES
        ):
            st.markdown(
                f'<div class="edr-export-choice"><b>{escape(label)}</b>'
                f"<span>{escape(purpose)}</span></div>",
                unsafe_allow_html=True,
            )
            if st.button(
                f"生成「{label}」",
                key=f"library_export_choice_{index}",
                use_container_width=True,
            ):
                _prepare_export(
                    database,
                    scope=choice_scope,
                    preset=choice_preset,
                    filters={
                        "journal": journal,
                        "abstract_state": abstract_state,
                        "decision": decision,
                        "auto_decision": auto_decision,
                        "year": year,
                        "search": search,
                    },
                    has_status=has_status,
                    library_scope=library_scope,
                    profile_id=profile_id,
                    label=label,
                )
        with st.expander("自定义导出范围"):
            scope = st.selectbox(
                "导出范围",
                list(LIBRARY_EXPORT_SCOPES),
                key="library_export_scope",
            )
            preset = st.selectbox(
                "导出内容",
                list(LIBRARY_EXPORT_PRESETS),
                key="library_export_preset",
                help="简洁版只含书目字段；阅读版增加摘要；研究版再加自动判断、人工决定与备注。",
            )
            if st.button("生成自定义导出文件", key="library_export_generate"):
                _prepare_export(
                    database,
                    scope=scope,
                    preset=preset,
                    filters={
                        "journal": journal,
                        "abstract_state": abstract_state,
                        "decision": decision,
                        "auto_decision": auto_decision,
                        "year": year,
                        "search": search,
                    },
                    has_status=has_status,
                    library_scope=library_scope,
                    profile_id=profile_id,
                    label=scope,
                )
        prepared = st.session_state.get("library_export_data")
        if prepared:
            csv_text, markdown_text, filename_base, count = prepared
            st.caption(f"已生成 {count} 篇论文的导出文件。")
            st.download_button("下载 RIS（Zotero / EndNote）", st.session_state["library_ris"],
                               file_name=f"{filename_base}.ris",
                               mime="application/x-research-info-systems")
            left, right = st.columns(2)
            left.download_button(
                "下载 CSV（表格）",
                csv_text,
                file_name=f"{filename_base}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            right.download_button(
                "下载 Markdown（纯文本）",
                markdown_text,
                file_name=f"{filename_base}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    for paper in page_papers:
        _render_paper_card(database, settings, paper, profile)


def _prepare_export(
    database: Database,
    *,
    scope: str,
    preset: str,
    filters: dict[str, str],
    has_status: bool,
    library_scope: LibraryScope,
    profile_id: str | None,
    label: str,
) -> None:
    """Materialise one export and remember it until the selection changes."""
    signature = (scope, preset, *filters.values())
    if st.session_state.get("library_export_signature") != signature:
        st.session_state.pop("library_export_data", None)
    st.session_state["library_export_signature"] = signature
    papers = _export_scope_papers(
        database,
        scope,
        filters=filters,
        has_status=has_status,
        library_scope=library_scope,
    )
    if not papers:
        st.session_state["library_export_data"] = None
        st.info(f"「{label}」在当前筛选条件下没有论文。")
        return
    st.session_state["library_ris"] = to_ris(papers, include_abstract=preset != "简洁版")
    rows = _preset_rows(papers, preset)
    prefix = (
        f"literature_radar_{profile_id}" if profile_id else "literature_radar"
    )
    filename_base = f"{prefix}_{scope}_{date.today().isoformat()}"
    st.session_state["library_export_data"] = (
        _rows_to_csv(rows),
        _rows_to_markdown(rows),
        filename_base,
        len(papers),
    )
