"""Local literature workbench (product path).

Phase 7B5: this page is the only product UI. It is driven by sync runs and
per-paper processing status and has no dependency on retired product mechanisms.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from eduecon_radar.services.batch_translation import BatchTitleTranslationService
    from eduecon_radar.services.translation import TranslationService

from eduecon_radar.config import Settings
from eduecon_radar.db import Database
from eduecon_radar.product import PRODUCT_NAME, PRODUCT_TAGLINE
from eduecon_radar.services import profile_store
from eduecon_radar.services.abstract_enrichment import (
    AbstractEnrichmentService,
)
from eduecon_radar.services.deepseek import BATCH_REQUEST_LIMIT, limit_batch_requests
from eduecon_radar.services.journal_registry import (
    ALL_GROUP as ALL_JOURNAL_GROUP,
)
from eduecon_radar.services.journal_registry import (
    CUSTOM_GROUP,
    JournalRegistry,
    RegistryJournal,
    make_custom_journal,
    normalize_journal_name,
    save_personal_journals,
)
from eduecon_radar.services.local_pipeline import (
    LocalPipelineServices,
    export_mentor_csv,
    export_mentor_markdown,
    export_research_csv,
    export_research_markdown,
    process_batch,
    rescreen_paper,
    resumable_paper_ids,
    retranslate_abstract,
    run_profile_id,
    save_human_decisions,
    status_label,
    sync_run_snapshot,
    translate_titles,
    translate_titles_for_paper_ids,
)
from eduecon_radar.services.migrations import ProductSafetyMigrationRequired
from eduecon_radar.services.paper_lifecycle_state import (
    screening_state_from_classification,
    screening_state_label,
)
from eduecon_radar.services.profile_registry import (
    CatalogJournal,
    ProductProfile,
    ProfileRegistry,
    load_profile_registry,
)
from eduecon_radar.services.profile_store import (
    list_runs_for_profile,
    profile_coverage,
)
from eduecon_radar.services.reading_desk import ReadingDesk, identity
from eduecon_radar.services.retrieval_verification import (
    VerificationResult,
    cached_verification,
)
from eduecon_radar.services.search_queue import SearchQueue
from eduecon_radar.services.source_traceability import (
    build_doi_url,
    resolve_paper_access,
)
from eduecon_radar.services.ui_labels import (
    CENTRALITY_VALUE_ZH,
    EVIDENCE_SUFFICIENCY_ZH,
    HUMAN_DECISION_OPTIONS,
    RELEVANCE_VALUE_ZH,
    human_decision_from_zh,
    zh_abstract_source,
    zh_auto_decision,
    zh_confidence,
    zh_human_decision,
    zh_screening_basis,
    zh_stop_reason,
    zh_validation_status,
)
from eduecon_radar.services.verification_intake import (
    CandidateIntakeState,
    candidate_intake_states,
    intake_verification_candidates,
)
from eduecon_radar.ui.theme import (
    StatusTone,
    render_page_header,
    render_section_header,
    render_status_badge,
    status_badge_html,
    workflow_progress_html,
)
from eduecon_radar.ui.views import daily_radar, paper_library

LOGGER = logging.getLogger(__name__)


def _processing_status_tone(status: str) -> StatusTone:
    if status in {"classified", "completed", "abstract_ready"}:
        return "complete"
    if status in {"failed", "needs_manual_abstract"}:
        return "failed"
    if status in {"needs_retry", "processing"}:
        return "retry"
    if status in {"pending", "classification_ready"}:
        return "pending"
    return "review"


def _auto_decision_tone(decision: object) -> StatusTone:
    value = str(decision or "")
    if value == "relevant":
        return "complete"
    if value in {"uncertain", "insufficient_evidence"}:
        return "review"
    if value in {"needs_abstract", "needs_retry"}:
        return "missing"
    if value == "irrelevant":
        return "info"
    return "pending"


def render(database: Database, settings: Settings) -> None:
    ai_configured = bool(getattr(settings, "deepseek_api_key", None))
    profile_registry = load_profile_registry(
        getattr(settings, "profiles_root", None),
        catalog_path=getattr(settings, "journal_catalog_path", None),
        default_profile_id=getattr(settings, "default_profile", "education_economics"),
    )
    profiles_ready = profile_store.profile_schema_ready(database)
    hosted = getattr(settings, "hosted", False)
    desk = ReadingDesk(database)
    if profiles_ready and "edr_profile" not in st.session_state:
        st.session_state["edr_profile"] = desk.get("active_profile")
    selected_profile = _selected_profile_id(profile_registry) if profiles_ready else (
        profile_store.DEFAULT_PROFILE_ID
    )
    profile = profile_registry.profile(selected_profile)
    if not profiles_ready:
        # Legacy mode: the database predates migration 0013.  It keeps working
        # exactly as before, but judgements are then run-scoped rather than
        # profile-scoped, so say so instead of failing silently.
        st.warning(
            "数据库尚未应用研究方向迁移（0013）：当前以单研究方向模式运行，"
            "判断仍按检索批次保存。备份后运行 eduecon-radar migrate-database "
            "--profile production --database <数据库路径> --apply 即可启用多研究方向。"
        )
    with st.container(key="edr_primary_navigation"):
        brand_col, nav_col, profile_col = st.columns([1.45, 1, 1], vertical_alignment="center")
        with brand_col:
            render_page_header(
                eyebrow="", title=PRODUCT_NAME, subtitle=PRODUCT_TAGLINE, heading="title",
            )
        with nav_col:
            views = (("今日雷达", "文献工作台", "论文库") if profiles_ready
                     else ("文献工作台", "论文库"))
            if target := st.session_state.pop("radar_nav", None):
                st.session_state["edr_primary_view"] = target
            active_view = (
                st.segmented_control(
                    "页面",
                    views,
                    default=views[0],
                    key="edr_primary_view",
                    label_visibility="collapsed",
                )
                or views[0]
            )
        with profile_col:
            choices = profile_registry.profile_choices() if profiles_ready else (
                (profile.profile_id, profile.display_name),
            )
            labels = {profile_id: name for profile_id, name in choices}
            chosen = st.selectbox(
                "研究方向",
                options=[profile_id for profile_id, _ in choices],
                index=[profile_id for profile_id, _ in choices].index(
                    profile.profile_id
                ),
                format_func=lambda value: labels.get(value, value),
                key="edr_profile_select",
                help="自动判断与人工决定按研究方向分别保存；论文元数据与摘要共用一份。",
            )
            if chosen != profile.profile_id:
                _reset_profile_scoped_state()
                desk.put("active_profile", chosen)
                st.session_state["edr_profile"] = chosen
                st.rerun()
    if not hosted and not ai_configured and active_view != "今日雷达":
        st.info(
            "尚未配置 DeepSeek API Key：可以检索、人工审核与导出；"
            "中文摘要与 AI 初筛需要先在根目录 .env 设置 DEEPSEEK_API_KEY。"
        )

    queue = None if hosted else SearchQueue(database)
    jobs = []
    if queue is not None:
        queue.start(settings)
        jobs = queue.active(profile.profile_id)
        if active_view != "今日雷达" and not jobs and (latest := queue.latest(profile.profile_id)):
            jobs = [latest]
    if active_view != "文献工作台":
        st.session_state.pop(f"waiting_search_{profile.profile_id}", None)
    overview = active_view == "今日雷达" and not st.session_state.get(
        "radar_focus_" + profile.profile_id
    )
    if overview or active_view == "文献工作台":
        for job in jobs:
            _render_search_activity(queue, profile.profile_id, job, overview=overview)
    if active_view == "论文库":
        paper_library.render(database, settings, profile=profile)
        return

    registry = profile.registry(extra_journals=_custom_journals(settings))
    if active_view == "今日雷达":
        with database.connect() as conn:
            last = conn.execute(
                "SELECT selected_journal_keys FROM sync_runs WHERE profile_id=? "
                "AND requested_limit IS NOT NULL ORDER BY id DESC LIMIT 1", (profile.profile_id,),
            ).fetchone()
        journals = {k: _resolve_selected_journal(registry, profile_registry, k)
                    for k in profile_registry.catalog}
        journals.update({j.journal_key: j for j in registry.journals if j.enabled})
        defaults = _recent_run_selection(
            [dict(last)] if last else [], registry, list(registry.groups), profile_registry
        )
        daily_radar.render(database, settings, profile, queue, journals, defaults)
        return
    runs = list_runs_for_profile(database, profile.profile_id)
    if runs and not hosted:
        run_labels = {
            int(run["id"]): (
                f"{run['query_start_date']} 至 {run['query_end_date']} · "
                f"{_local_started_at(run['started_at'])} 获取 · #{run['id']}"
            )
            for run in runs
        }
        ids = list(run_labels)
        picker_key = f"batch_choice_{profile.profile_id}"
        new_id = st.session_state.pop("just_fetched_run", None)
        if new_id in ids:
            st.session_state[picker_key] = new_id
        elif picker_key not in st.session_state and st.session_state.get("local_run_id") in ids:
            st.session_state[picker_key] = st.session_state["local_run_id"]
        st.session_state["local_run_id"] = st.selectbox(
            "检索记录", ids,
            format_func=run_labels.get, key=picker_key,
        )
    snapshot = _selected_snapshot(database, runs)
    if not hosted:
        with st.expander("新建检索", expanded=snapshot is None):
            _render_retrieval(
                database, settings, registry, runs, snapshot, profile, profile_registry
            )
    elif not snapshot:
        st.info("本方向暂无阅读批次。可在论文库选择「全部方向」查找论文并保存判断。")
    if not snapshot:
        return
    if message := st.session_state.pop("retrieval_notice", None):
        st.info(message)
    if snapshot["run"].get("status") in {"partial", "failed"}:
        st.warning("部分期刊检索异常，本批可能有遗漏。可在「新建检索」中重试同一范围。")
    if not hosted:
        _render_workbench_status(snapshot, profile, _next_action(snapshot))
    run_id = int(snapshot["run"]["id"])
    stage_key = f"batch_stage_{run_id}"
    if st.session_state.get(f"local_review_target_{run_id}"):
        st.session_state[stage_key] = "逐篇审核"
    stage = st.segmented_control(
        "本批论文",
        ("逐篇审核", "导出") if hosted else ("逐篇审核", "补全与重试", "导出", "检索核验"),
        default="逐篇审核", key=stage_key, label_visibility="collapsed",
    )
    if stage == "补全与重试" and any(
        j["run_id"] == run_id and j["phase"] == "analysis" for j in jobs
    ):
        st.info("正在后台补全和分析本批论文。你可以直接阅读和保存人工决定。")
    elif stage == "补全与重试":
        _render_processing(database, settings, snapshot, profile)
    elif stage == "导出":
        _render_exports(database, snapshot, allow_archive=not hosted)
    elif stage == "检索核验":
        with limit_batch_requests():
            _render_verification_summary_section(database, settings, registry, snapshot, profile)
        _render_acquired_detail(snapshot)
    else:
        _render_review(database, settings, snapshot, profile)


def _local_started_at(value: object) -> str:
    try:
        timestamp = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value or "")
    return timestamp.astimezone().strftime("%Y-%m-%d %H:%M")


def _selected_profile_id(registry: object) -> str:
    stored = str(st.session_state.get("edr_profile") or "").strip()
    available = {profile_id for profile_id, _ in registry.profile_choices()}
    if stored in available:
        return stored
    default = str(getattr(registry, "default_profile_id", "education_economics"))
    st.session_state["edr_profile"] = default
    return default


#: Widget state that only means something inside one profile: journal groups,
#: the selected journals and the library filters all reference profile-specific
#: names.  Carrying them across a profile switch shows the wrong list or raises
#: on an unknown group key, so they are cleared on an explicit switch.
_PROFILE_SCOPED_WIDGET_KEYS = (
    "local_journal_group",
    "local_journal_scope",
    "local_journal_search",
    "local_journal_subject",
    "local_journal_picker",
    "local_journal_picker_scope",
    "local_selected_journal_keys",
    "library_journal",
    "library_page",
    "library_export_signature",
    "library_export_data",
    "local_run_id",
)
_PROFILE_SCOPED_WIDGET_PREFIXES = ("local_journals_",)


def _reset_profile_scoped_state() -> None:
    for key in list(st.session_state):
        name = str(key)
        if name in _PROFILE_SCOPED_WIDGET_KEYS or name.startswith(
            _PROFILE_SCOPED_WIDGET_PREFIXES
        ):
            st.session_state.pop(key, None)


def _custom_journals(settings: Settings) -> tuple[RegistryJournal, ...]:
    """Locally added journals, kept out of the global catalog."""
    path = getattr(settings, "personal_journals_path", None)
    if path is None or not Path(path).is_file():
        return ()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = []
    for item in payload.get("journals") or []:
        entries.append(
            RegistryJournal(
                journal_key=str(item["journal_key"]),
                canonical_name=str(item["canonical_name"]),
                short_name=str(item["short_name"]),
                aliases=tuple(str(value) for value in item.get("aliases") or ()),
                print_issn=str(item.get("print_issn") or ""),
                online_issn=item.get("online_issn") or None,
                publisher=str(item.get("publisher") or ""),
                journal_group=CUSTOM_GROUP,
                subject_tags=tuple(str(value) for value in item.get("subject_tags") or ()),
                enabled=bool(item.get("enabled", True)),
                display_order=int(item.get("display_order") or 999),
                retrieval_priority=int(item.get("retrieval_priority") or 4),
                metadata_source="personal_config",
                registry_version=str(payload.get("registry_version") or ""),
            )
        )
    return tuple(entries)


def _selected_snapshot(
    database: Database,
    runs: list[dict[str, object]],
) -> dict[str, object] | None:
    if not runs:
        return None
    run_ids = [int(run["id"]) for run in runs]
    selected = st.session_state.get("local_run_id")
    run_id = selected if selected in run_ids else run_ids[0]
    run = next(item for item in runs if int(item["id"]) == run_id)
    return sync_run_snapshot(database, run_id, run=run)


def _render_flow(snapshot: dict[str, object] | None) -> None:
    counts = snapshot["counts"] if snapshot else {}
    st.markdown(
        workflow_progress_html(
            (
                ("本次获取", counts.get("retrieved", 0), "info"),
                ("待补摘要", counts.get("missing_abstract", 0), "missing"),
                ("待审核", counts.get("review_pending", 0), "review"),
                ("已完成", counts.get("completed", 0), "complete"),
            )
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="edr-flow-note">以上为「本次获取」这一批论文的数量；'
        "论文库包含全部研究方向的长期记录。</div>",
        unsafe_allow_html=True,
    )


@dataclass(frozen=True)
class NextAction:
    """What the researcher should do next, in one sentence.

    Derived from the same snapshot counts the workflow line shows, so the hint
    and the numbers can never disagree.  It names the action only; the buttons
    stay where their task lives, so no second control path is introduced.
    """

    headline: str
    detail: str
    tone: StatusTone


def _next_action(snapshot: dict[str, object]) -> NextAction:
    counts = snapshot.get("counts") or {}
    pending_review = int(counts.get("review_pending", 0) or 0)
    missing_abstract = int(counts.get("missing_abstract", 0) or 0)
    unfinished = int(counts.get("pending_processing", 0) or 0) + int(
        counts.get("needs_retry", 0) or 0
    )
    if counts.get("retrieved") == 0:
        return NextAction("本批没有论文", "可调整期刊或日期重新检索，或查看检索核验。", "info")
    if counts.get("retrieved") and counts.get("completed") == counts["retrieved"]:
        return NextAction("本批已全部审核", "可导出阅读清单，或新建下一次检索。", "complete")
    if pending_review:
        return NextAction(
            f"下一步：审核 {pending_review} 篇待判论文",
            "逐篇查看中文摘要与 AI 建议，明确记录保留或排除。",
            "review",
        )
    if unfinished:
        return NextAction(
            f"下一步：处理 {unfinished} 篇未完成论文",
            "在「补全与重试」中继续补全摘要、翻译与初筛。",
            "pending",
        )
    if missing_abstract:
        return NextAction(
            f"下一步：处理 {missing_abstract} 篇缺摘要论文",
            "可重试数据源，或在审核页粘贴摘要后重新分析。",
            "missing",
        )
    return NextAction(
        "本批已全部完成审核",
        "可以继续检索新论文，或在论文库中筛选并导出。",
        "complete",
    )


def _render_workbench_status(
    snapshot: dict[str, object],
    profile: ProductProfile,
    action: NextAction,
) -> None:
    """The daily glance: which queue, how much is in it, and what comes next.

    Everything here is a count the researcher acts on today.  Retrieval
    internals (provider page counts, stop reasons, cache state) stay in the
    collapsed run detail instead of competing for this position.
    """
    counts = snapshot.get("counts") or {}
    reviewed = sum(
        1 for paper in snapshot["papers"] if paper.get("human_decision") != "undecided"
    )
    retrieved = int(counts.get("retrieved", 0) or 0)
    metrics = (
        (str(retrieved), "本次获取"),
        (str(int(counts.get("review_pending", 0) or 0)), "待审核"),
        (str(int(counts.get("missing_abstract", 0) or 0)), "待补摘要"),
        (f"{reviewed}/{retrieved}" if retrieved else "0", "已给决定"),
    )
    run = snapshot["run"]
    period = f"{run.get('query_start_date')} 至 {run.get('query_end_date')}"
    st.markdown(
        '<div class="edr-status-bar">'
        '<div class="edr-status-profile">'
        f'<span class="edr-status-label">当前研究方向</span>'
        f'<span class="edr-status-name">{escape(profile.display_name)}</span>'
        f'<span class="edr-status-period">{escape(period)}</span>'
        "</div>"
        '<div class="edr-status-metrics">'
        + "".join(
            f'<div class="edr-status-metric"><b>{escape(value)}</b>'
            f"<span>{escape(label)}</span></div>"
            for value, label in metrics
        )
        + "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="edr-next-action edr-next-action--{action.tone}">'
        f'<b>{escape(action.headline)}</b>'
        f"<span>{escape(action.detail)}</span></div>",
        unsafe_allow_html=True,
    )


def _render_retrieval(
    database: Database,
    settings: Settings,
    registry: JournalRegistry,
    runs: list[dict[str, object]],
    snapshot: dict[str, object] | None,
    profile: ProductProfile | None = None,
    profile_registry: ProfileRegistry | None = None,
) -> None:
    with st.container(key="edr_retrieval_panel"):
        st.markdown(
            '<div class="edr-hero-eyebrow">今日检索</div>',
            unsafe_allow_html=True,
        )
        st.header("获取并分析论文")
        _render_retrieval_controls(
            database,
            settings,
            registry,
            runs,
            profile=profile,
            profile_registry=profile_registry,
        )
        if snapshot is None:
            _render_flow(snapshot)


def _profile_default_journal_keys(
    registry: JournalRegistry, group_keys: list[str]
) -> list[str]:
    """The profile's primary journal group, used as the default selection."""
    if not group_keys:
        return []
    return [journal.journal_key for journal in registry.expand_group(group_keys[0])]


def _recent_run_selection(
    runs: list[dict[str, object]],
    registry: JournalRegistry,
    group_keys: list[str],
    profile_registry: ProfileRegistry | None = None,
) -> tuple[str, ...]:
    """Journals to preselect, in the user's own last-used order.

    ``selected_journal_keys`` is stored per run and may be either JSON text or an
    already-decoded list; a selection carried over from another profile is
    dropped rather than silently producing an empty journal list.
    """
    raw = runs[0].get("selected_journal_keys") if runs else None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "[]")
        except ValueError:
            raw = None
    keys = [str(key) for key in (raw or [])] if isinstance(raw, list) else []
    known = {
        journal.journal_key
        for group in group_keys
        for journal in registry.expand_group(group)
    }
    if profile_registry is not None:
        known.update(profile_registry.catalog)
    carried = tuple(key for key in keys if key in known)
    if carried:
        return carried
    return tuple(_profile_default_journal_keys(registry, group_keys))


def _render_retrieval_controls(
    database: Database,
    settings: Settings,
    registry: JournalRegistry,
    runs: list[dict[str, object]],
    profile: ProductProfile | None = None,
    profile_registry: object | None = None,
) -> None:
    """Journals are chosen from one global library, recommended by the profile.

    A direction defines *tiers* (its core journals, its adjacent journals, the
    shared flagship list) rather than owning journals, so the same QJE or JOLE
    exists once and can be recommended by several queues.  A researcher can
    always step outside the recommendation and browse or search the whole
    catalog, which is why "全部期刊" and a subject filter exist here.
    """
    group_keys = list(registry.groups.keys())
    left_col, right_col = st.columns([1.12, 0.88], vertical_alignment="top")
    with left_col:
        selected, custom = _render_journal_selector(
            settings,
            registry,
            runs,
            profile=profile,
            profile_registry=profile_registry,
            group_keys=group_keys,
        )
    if selected:
        st.session_state["local_selected_journal_keys"] = list(selected)
    # A journal chosen from the global library may sit outside this profile's
    # recommendations, so it is resolved against the full catalog and simply
    # falls outside the profile's ranking.  The profile describes what it
    # *recommends*, not what it permits.
    chosen_list: list[RegistryJournal] = []
    for key in selected:
        journal = _resolve_selected_journal(registry, profile_registry, key)
        if journal is not None:
            chosen_list.append(journal)
    chosen = tuple(chosen_list)
    if custom is not None:
        chosen = (*chosen, custom)
    with right_col, st.form("search_dates_" + (profile.profile_id if profile else "default")):
        date_cols = st.columns(2)
        since = date_cols[0].date_input(
            "起始日期",
            value=date.today() - timedelta(days=90),
            min_value=date(1600, 1, 1),
            key="local_since",
        )
        until = date_cols[1].date_input(
            "结束日期", value=date.today(), min_value=date(1600, 1, 1), key="local_until"
        )
        analyze = st.checkbox(
            "获取后自动翻译与初筛", value=bool(getattr(settings, "deepseek_api_key", None)),
            disabled=not getattr(settings, "deepseek_api_key", None), key="local_analyze_on_fetch",
        )
        if analyze:
            st.caption(f"本批自动分析最多 {BATCH_REQUEST_LIMIT} 次请求（含重试），保留全部论文。")
        submitted = st.form_submit_button(
            "获取论文",
            type="primary",
            disabled=not chosen,
            key="local_fetch",
            use_container_width=True,
        )
    if submitted:
        if until < since:
            st.error("结束日期不能早于起始日期。")
            return
        queue = SearchQueue(database)
        direction = profile.profile_id if profile else profile_store.DEFAULT_PROFILE_ID
        job_id = queue.submit(direction, chosen, since, until, analyze)
        st.session_state[f"waiting_search_{direction}"] = job_id
        queue.start(settings)
        st.rerun()


def _render_search_activity(queue, profile_id, initial, *, overview=False):
    interval = 4 if initial["phase"] in {"fetch", "analysis"} else None
    if initial["phase"] == "abstracts":
        interval = 60
    @st.fragment(run_every=interval)
    def activity():
        job = queue.status(initial["id"])
        if not job:
            return
        waiting = f"waiting_search_{profile_id}"
        if job["id"] == st.session_state.get(waiting) and job["run_id"]:
            st.session_state.pop(waiting)
            st.session_state["just_fetched_run"] = job["run_id"]
            st.rerun()
        if overview and job["phase"] != initial["phase"]:
            st.rerun()
        if job["phase"].startswith(("attention:", "paused:")):
            st.caption("已暂停" if job["phase"].startswith("paused:") else job["progress"])
            if st.button("继续处理", key=f"recover_search_{job['id']}"):
                queue.restart(job["id"])
                st.rerun()
        elif job["phase"] in {"fetch", "analysis"}:
            info, control = st.columns([5, 1])
            info.caption(job["progress"])
            if control.button("暂停处理", key=f"pause_search_{job['id']}"):
                queue.pause(job["id"])
                st.rerun()
        else:
            st.caption(job["progress"])
            if job["run_id"] and initial["phase"] in {"analysis", "fetch"} and st.button(
                "查看处理结果", key=f"open_result_{job['id']}"
            ):
                st.session_state["just_fetched_run"] = job["run_id"]
                st.rerun()

    activity()


#: Scope labels for the journal selector.  The first two are the current
#: direction's own recommendations; the third is the shared flagship list; the
#: fourth opens the whole catalog.  No label names a discipline, so switching
#: profile renames nothing.
JOURNAL_SCOPE_CORE = "本方向核心"
JOURNAL_SCOPE_ADJACENT = "本方向相关"
JOURNAL_SCOPE_RECOMMENDED = "本方向全部推荐"
JOURNAL_SCOPE_LIBRARY = "全部期刊库"
JOURNAL_SCOPE_CUSTOM = "自定义期刊"
JOURNAL_SCOPES = (
    JOURNAL_SCOPE_CORE,
    JOURNAL_SCOPE_ADJACENT,
    JOURNAL_SCOPE_RECOMMENDED,
    JOURNAL_SCOPE_LIBRARY,
    JOURNAL_SCOPE_CUSTOM,
)


def _profile_tier_keys(registry: JournalRegistry, group_keys: list[str]) -> dict[str, list[str]]:
    """The profile's tiers, split into core / adjacent / the rest.

    Tiers are read from the profile's own group keys, so a profile with a
    different tier layout (political economy has two adjacent groups) needs no
    special case here.
    """
    core: list[str] = []
    adjacent: list[str] = []
    other: list[str] = []
    for key in group_keys:
        if key == CUSTOM_GROUP:
            continue
        journals = [item.journal_key for item in registry.expand_group(key)]
        lowered = key.casefold()
        if "core" in lowered:
            core.extend(journals)
        elif "adjacent" in lowered:
            adjacent.extend(journals)
        else:
            other.extend(journals)
    return {
        "core": list(dict.fromkeys(core)),
        "adjacent": list(dict.fromkeys(adjacent)),
        "other": list(dict.fromkeys(other)),
    }


def _render_journal_selector(
    settings: Settings,
    registry: JournalRegistry,
    runs: list[dict[str, object]],
    *,
    profile: ProductProfile | None,
    profile_registry: ProfileRegistry | None,
    group_keys: list[str],
) -> tuple[list[str], RegistryJournal | None]:
    """Scope + optional subject filter + searchable picker over one library."""
    tiers = _profile_tier_keys(registry, group_keys)
    adjacent = [*tiers["core"], *tiers["adjacent"]]
    recommended = [*adjacent, *tiers["other"]]

    stored_scope = str(st.session_state.get("local_journal_scope") or "")
    scope = st.radio(
        "期刊范围",
        JOURNAL_SCOPES,
        index=JOURNAL_SCOPES.index(stored_scope) if stored_scope in JOURNAL_SCOPES else 0,
        horizontal=True,
        key="local_journal_scope",
        label_visibility="collapsed",
    )

    subject_filter = ""
    if scope == JOURNAL_SCOPE_LIBRARY:
        subjects = (
            profile_registry.catalog_subjects() if profile_registry is not None else ()
        )
        options = ["全部领域", *[f"{name}（{count}）" for _, name, count in subjects]]
        if "local_journal_subject" not in st.session_state:
            st.session_state["local_journal_subject"] = options[0]
        if st.session_state.get("local_journal_subject") not in options:
            st.session_state["local_journal_subject"] = options[0]
        chosen_subject = st.selectbox("领域", options, key="local_journal_subject")
        if chosen_subject != options[0]:
            subject_filter = chosen_subject.rsplit("（", 1)[0]

    if scope == JOURNAL_SCOPE_CUSTOM:
        return [], _render_custom_journal(settings, registry)

    query = st.text_input(
        "搜索期刊",
        key="local_journal_search",
        placeholder="期刊名称、简称或 ISSN",
    )

    if scope == JOURNAL_SCOPE_CORE:
        pool = [registry.by_key(key) for key in tiers["core"]]
        description = "当前研究方向最核心的期刊。"
    elif scope == JOURNAL_SCOPE_ADJACENT:
        pool = [registry.by_key(key) for key in adjacent]
        description = "本方向核心期刊，加上经常发表该方向研究的相关期刊。"
    elif scope == JOURNAL_SCOPE_RECOMMENDED:
        pool = [registry.by_key(key) for key in recommended]
        description = "本方向的全部推荐期刊，含所有研究方向共用的综合与旗舰期刊。"
    else:
        pool = [
            _journal_for_catalog_entry(registry, entry)
            for entry in sorted(
                _catalog_entries(profile_registry, profile),
                key=lambda item: (item.subject, item.canonical_name),
            )
        ]
        description = (
            "全局期刊库：可以检索并选择任何领域的经济学期刊。"
            "当前研究方向的初筛提示词不变，人工审核后仍按本方向保存决定。"
        )

    if subject_filter and scope == JOURNAL_SCOPE_LIBRARY and profile_registry is not None:
        allowed = {
            entry.journal_key
            for entry in _catalog_entries(profile_registry, profile)
            if entry.subject == subject_filter
            or profile_registry.catalog_subject_names.get(entry.subject) == subject_filter
        }
        pool = [journal for journal in pool if journal.journal_key in allowed]

    if query:
        matched = {journal.journal_key for journal in registry.search(query)}
        if profile_registry is not None:
            needle = normalize_journal_name(query)
            matched |= {
                entry.journal_key
                for entry in _catalog_entries(profile_registry, profile)
                if needle
                and any(
                    needle in normalize_journal_name(value)
                    for value in (
                        entry.canonical_name,
                        entry.short_name,
                        *entry.aliases,
                        *entry.issns,
                    )
                )
            }
        pool = [journal for journal in pool if journal.journal_key in matched]

    st.caption(f"{description} 当前可选 {len(pool)} 本。")

    available_keys = [journal.journal_key for journal in pool]
    if "local_selected_journal_keys" not in st.session_state:
        st.session_state["local_selected_journal_keys"] = list(
            _recent_run_selection(runs, registry, group_keys, profile_registry)
        )

    selected_keys = list(st.session_state["local_selected_journal_keys"])
    choices = {journal.journal_key: journal for journal in pool}
    for key in selected_keys:
        journal = _resolve_selected_journal(registry, profile_registry, key)
        if journal:
            choices.setdefault(key, journal)
    picker = "local_journal_picker"
    action_cols = st.columns(3)
    if action_cols[0].button("全选当前列表", key="local_journal_select_all"):
        selected_keys = list(dict.fromkeys([*selected_keys, *available_keys]))
    if action_cols[1].button("清除选择", key="local_journal_clear"):
        selected_keys = []
    if action_cols[2].button("恢复本方向推荐", key="local_journal_reset"):
        selected_keys = _profile_default_journal_keys(registry, group_keys)
        choices.update({key: registry.by_key(key) for key in selected_keys})

    def remember_selection() -> None:
        st.session_state["local_selected_journal_keys"] = list(st.session_state[picker])

    st.session_state[picker] = [key for key in selected_keys if key in choices]
    selected = st.multiselect(
        "已选期刊", list(choices),
        format_func=lambda key: _journal_label(choices[key]),
        key=picker, on_change=remember_selection,
    )
    st.session_state["local_selected_journal_keys"] = list(selected)
    return list(selected), None


def _resolve_selected_journal(
    registry: JournalRegistry,
    profile_registry: ProfileRegistry | None,
    key: str,
) -> RegistryJournal | None:
    """Resolve a picked journal key to the shape retrieval expects.

    First the profile's own registry (which carries its ranking), then the
    global catalog for a journal the profile does not recommend.  An unknown key
    is dropped rather than raising, so a stale session cannot break the page.
    """
    try:
        return registry.by_key(key)
    except KeyError:
        pass
    if profile_registry is None:
        return None
    entry = profile_registry.catalog.get(key)
    if entry is None:
        return None
    return _journal_for_catalog_entry(registry, entry)


def _catalog_entries(
    profile_registry: ProfileRegistry | None, profile: ProductProfile | None
) -> tuple[CatalogJournal, ...]:
    """Every journal in the global catalog, as catalog entries.

    Read through the profile registry so the UI has one source for the library
    and cannot drift from what the profiles reference.  ``profile`` is accepted
    for call-site symmetry; the library is global and therefore profile-independent.
    """
    del profile
    if profile_registry is None:
        return ()
    return tuple(profile_registry.catalog.values())


def _journal_for_catalog_entry(
    registry: JournalRegistry, entry: CatalogJournal
) -> RegistryJournal:
    """Adapt a catalog entry to the shape the picker uses.

    The profile's registry only holds the journals that profile monitors; the
    library scope has to reach beyond it, so an entry outside the profile is
    presented through the same interface without being added to the profile.
    """
    key = entry.journal_key
    try:
        return registry.by_key(key)
    except KeyError:
        return RegistryJournal(
            journal_key=key,
            canonical_name=entry.canonical_name,
            short_name=entry.short_name,
            aliases=tuple(entry.aliases),
            print_issn=entry.print_issn,
            online_issn=entry.online_issn,
            publisher=entry.publisher,
            journal_group=ALL_JOURNAL_GROUP,
            subject_tags=tuple(entry.subject_tags),
            enabled=True,
            display_order=0,
            retrieval_priority=entry.default_priority,
            metadata_source=entry.metadata_source,
            registry_version=entry.catalog_version,
        )


def _render_custom_journal(settings: Settings, registry: JournalRegistry) -> RegistryJournal | None:
    with st.expander("自定义期刊", expanded=True):
        name = st.text_input("期刊名称", key="local_custom_name")
        short = st.text_input("简称", key="local_custom_short")
        print_issn = st.text_input("ISSN", key="local_custom_issn")
        online_issn = st.text_input("Online ISSN（可选）", key="local_custom_eissn")
        if not (name and short and print_issn):
            return None
        try:
            parsed = make_custom_journal(
                registry,
                canonical_name=name,
                short_name=short,
                print_issn=print_issn,
                online_issn=online_issn or None,
            )
        except ValueError as exc:
            st.error(str(exc))
            return None
        if st.button("保存到本地期刊配置", key="save_local_custom_journal"):
            save_personal_journals(
                getattr(settings, "personal_journals_path", None)
                or settings.project_root / "data" / "development" / "personal_journals.json",
                (parsed,),
            )
            st.success("已保存到本地期刊配置。")
        return parsed


def _render_processing(
    database: Database,
    settings: Settings,
    snapshot: dict[str, object],
    profile: ProductProfile | None = None,
) -> None:
    """Batch processing: one action, plus the record of how the chain went.

    The per-status counters used to be printed three times on this page (in the
    workflow line, as a section meta, and as a badge row).  What is left is the
    decision the researcher actually makes — continue processing or not — and an
    optional detail view for when a specific paper looks stuck.
    """
    counts = snapshot.get("counts") or {}
    meta = " · ".join(
        f"{label} {counts.get(key, 0)}"
        for label, key in (
            ("待处理", "pending_processing"),
            ("需重试", "needs_retry"),
            ("英文摘要", "english_abstracts"),
            ("中文摘要", "chinese_abstracts"),
            ("自动判断", "automatic_judgments"),
        )
    )
    render_section_header("处理进度", meta=meta)
    if profile is not None:
        coverage = profile_coverage(database, profile.profile_id)
        st.caption(
            f"研究方向「{profile.display_name}」累计 {coverage['papers']} 篇论文 · "
            f"自动判断 {coverage['judged']} · 人工决定 {coverage['decided']}"
        )
    st.caption(f"每次批量处理最多 {BATCH_REQUEST_LIMIT} 次模型请求，含重试。")
    if message := st.session_state.pop(f"processing_notice_{snapshot['run']['id']}", None):
        st.info(message)
    with st.container(key="edr_processing_panel"), limit_batch_requests():
        _render_processing_controls(database, settings, snapshot, profile)


def _render_processing_controls(
    database: Database,
    settings: Settings,
    snapshot: dict[str, object],
    profile: ProductProfile | None = None,
) -> None:
    run_id = int(snapshot["run"]["id"])
    papers = snapshot["papers"]
    counts = snapshot.get("counts") or {}
    pending = int(counts.get("pending_processing", 0) or 0) + int(
        counts.get("needs_retry", 0) or 0
    )
    if pending:
        st.markdown(
            f'<p class="edr-action-hint">还有 {pending} 篇论文没有完成'
            "摘要、翻译与初筛；处理过程逐篇进行，单篇失败不影响其他论文。</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="edr-action-hint">本批没有等待自动续跑的论文，'
            "可以直接进入人工审核。</p>",
            unsafe_allow_html=True,
        )
    if pending:
        st.caption("每次继续最多 10 篇，优先处理尚未尝试的论文。")
    if st.button(
        "处理剩余论文",
        type="primary",
        key=f"local_process_{run_id}",
        disabled=not pending,
        use_container_width=True,
    ):
        if not getattr(settings, "deepseek_api_key", None):
            st.error("尚未配置DeepSeek API Key。请在根目录 .env 设置 DEEPSEEK_API_KEY。")
        else:
            # Same definition of "unfinished" as the CLI continuation command;
            # papers left in an intermediate stage must stay reachable here too.
            pending_ids = resumable_paper_ids(database, run_id)[:10]
            if not pending_ids:
                st.info("当前没有待处理的论文。")
                return
            try:
                with st.status("正在处理剩余论文", expanded=True) as progress:
                    completed: list[int] = []
                    needs_abstract: list[int] = []
                    needs_retry: list[int] = []
                    failed: list[int] = []
                    ai_calls = 0
                    services = _build_services(database, settings, profile)
                    progress.update(label="正在补译待处理论文的标题…")
                    translate_titles_for_paper_ids(database, run_id, pending_ids, services)
                    for index, paper_id in enumerate(pending_ids, 1):
                        progress.update(
                            label=f"正在处理第 {index}/{len(pending_ids)} 篇（摘要、翻译、初筛）…"
                        )
                        try:
                            partial = process_batch(
                                database,
                                run_id,
                                (paper_id,),
                                services,
                            )
                        except Exception:
                            LOGGER.exception(
                                "单篇处理失败 run_id=%s paper_id=%s",
                                run_id,
                                paper_id,
                            )
                            failed.append(paper_id)
                        else:
                            completed.extend(partial.get("completed") or [])
                            needs_abstract.extend(partial.get("needs_abstract") or [])
                            needs_retry.extend(partial.get("needs_retry") or [])
                            ai_calls += int(partial.get("ai_calls") or 0)
                    progress.update(
                        label=(
                            f"完成 {len(completed)} 篇；"
                            f"待补摘要 {len(needs_abstract)} 篇；"
                            f"需重试 {len(needs_retry)} 篇"
                            + (f"；失败 {len(failed)} 篇" if failed else "")
                        ),
                        state="complete",
                    )
            except Exception as exc:
                _render_operation_error("论文处理失败", exc, stage="processing")
            else:
                st.session_state[f"processing_notice_{run_id}"] = (
                    f"本次处理完成 {len(completed)} 篇；"
                    f"待补摘要 {len(needs_abstract)} 篇；"
                    f"需重试 {len(needs_retry)} 篇；"
                    + (f"失败 {len(failed)} 篇；" if failed else "")
                    + f"AI 调用 {ai_calls}。"
                )
                st.rerun()
    with st.expander("维护与重试"):
        title_col, abstract_col = st.columns(2)
        if title_col.button(
            "补译缺失标题",
            key=f"local_translate_missing_titles_{run_id}",
            use_container_width=True,
        ):
            try:
                result = translate_titles(
                    database,
                    run_id,
                    _build_services(database, settings, profile),
                )
            except Exception as exc:
                _render_operation_error("标题补译失败", exc, stage="title_backfill")
            else:
                st.session_state[f"processing_notice_{run_id}"] = (
                    f"已补译 {result['translated']} 篇；"
                    f"待处理 {result['pending']} 篇；失败 {result['failed']} 篇。"
                )
                st.rerun()
        if abstract_col.button(
            "重试缺失摘要",
            key=f"local_retry_missing_abstracts_{run_id}",
            use_container_width=True,
        ):
            missing_ids = tuple(
                int(paper["id"])
                for paper in papers
                if not paper.get("abstract")
                and dict(dict(paper.get("processing") or {}).get("abstract_retrieval") or {}).get(
                    "status"
                )
                != "not_applicable"
            )
            if not missing_ids:
                st.info("本次获取的论文均已有英文摘要。")
            else:
                try:
                    with st.spinner("正在重试摘要数据源……"):
                        result = process_batch(
                            database,
                            run_id,
                            missing_ids[:10],
                            _build_services(database, settings, profile),
                            force_abstract_refresh=True,
                        )
                except Exception as exc:
                    _render_operation_error(
                        "本次摘要重试失败",
                        exc,
                        stage="abstract_retrieval_retry_batch",
                    )
                else:
                    st.session_state[f"processing_notice_{run_id}"] = (
                        f"完成 {len(result['completed'])} 篇；"
                        f"仍需人工补充摘要 {len(result['needs_abstract'])} 篇。"
                    )
                    st.rerun()


def _condition_rows(classification: dict[str, object]) -> list[tuple[str, str]]:
    """The named conditions the screening call reported, in plain language.

    The stored keys (``education_centrality`` / ``economic_relevance``) are
    historical; their *meaning* is profile content, so the centrality row is
    labelled with the profile's own dimension name.
    """
    return [
        ("判断依据", zh_screening_basis(classification.get("screening_basis"))),
        ("中心性", _lookup(CENTRALITY_VALUE_ZH, "education_centrality", classification)),
        ("经济学内容", _lookup(RELEVANCE_VALUE_ZH, "economic_relevance", classification)),
        ("证据是否充分", _lookup(EVIDENCE_SUFFICIENCY_ZH, "evidence_sufficiency", classification)),
        ("自动校验", zh_validation_status(classification.get("validation_status"))),
    ]


def _lookup(
    labels: dict[str, str],
    key: str,
    classification: dict[str, object],
) -> str:
    return labels.get(str(classification.get(key) or "").strip(), "")


def _ai_reasoning_block(
    classification: dict[str, object],
    evidence: list[object],
    *,
    has_abstract: bool,
    title: str = "",
    abstract: str = "",
) -> str:
    """'Why did the AI say this?' — the audit record inside the disclosure.

    Kept inside the review page but no longer on the decision surface: a
    reviewer decides from the proposed number and how much evidence stood behind
    it, and only opens this when a call is actually being challenged.  It holds
    the model's reason, the words it was based on, how much text it saw, the
    four named conditions and the validator's verdict, so a verdict can still be
    reproduced; the raw JSON stays in the record below it.
    """
    confidence = zh_confidence(classification.get("confidence"))
    rows = [
        f'<div class="edr-reason-row"><b>{escape(label)}</b>'
        f"<span>{escape(value)}</span></div>"
        for label, value in _condition_rows(classification)
        if value
    ]
    if confidence:
        rows.insert(
            0,
            '<div class="edr-reason-row"><b>置信度</b>'
            f'<span>{escape(confidence)}</span></div>',
        )
    reason = str(classification.get("reason") or "").strip()
    reason_html = (
        f'<p class="edr-reason-text">{escape(reason)}</p>'
        if reason
        else '<p class="edr-reason-text edr-reason-text--quiet">'
        "本次判断没有记录理由。</p>"
    )
    quotes = [
        str(quote).strip()
        for quote in evidence
        if str(quote or "").strip() and str(quote).strip().casefold() != "none"
    ]
    if quotes:
        title = " ".join(title.casefold().split())
        abstract = " ".join(abstract.casefold().split())
        lines = []
        for quote in quotes[:6]:
            snippet = " ".join(quote.casefold().split())
            source = "未匹配当前原文"
            if snippet in abstract:
                source = "摘要"
            elif snippet in title:
                source = "标题"
            lines.append(f"<li><small>{source}</small> · {escape(quote)}</li>")
        quotes_html = (
            '<div class="edr-reason-quotes"><b>原文核对</b><ul>' + "".join(lines) + "</ul></div>"
        )
    elif has_abstract:
        quotes_html = (
            '<div class="edr-reason-quotes edr-reason-quotes--quiet">'
            "本次判断未引用具体原文片段。</div>"
        )
    else:
        # Saying "no quotes" without saying why reads like a defect.  There is
        # no abstract to quote, which is exactly what the basis row records.
        quotes_html = (
            '<div class="edr-reason-quotes edr-reason-quotes--quiet">'
            "尚无可引用的英文摘要原文，判断依据见上。</div>"
        )
    return (
        '<div class="edr-reason">'
        '<div class="edr-reason-title">AI 判断依据</div>'
        f"{reason_html}"
        f'<div class="edr-reason-grid">{"".join(rows)}</div>'
        f"{quotes_html}</div>"
    )


def _render_review(
    database: Database,
    settings: Settings,
    snapshot: dict[str, object],
    profile: ProductProfile | None = None,
) -> None:
    render_section_header("人工审核")
    with st.container(key="edr_review_workspace"):
        _render_review_workspace(database, settings, snapshot, profile)


def _next_unread(papers: list[dict[str, object]], current: int = -1) -> int | None:
    """Show promising and uncertain papers first. Every undecided paper stays reachable."""
    candidates = []
    for offset in range(1, len(papers) + 1):
        index = (current + offset) % len(papers)
        paper = papers[index]
        if index == current or paper.get("human_decision") not in {None, "", "undecided"}:
            continue
        decision = (paper.get("processing") or {}).get("classification", {}).get("decision")
        priority = {"relevant": 0, "uncertain": 1, "irrelevant": 3}.get(decision, 2)
        candidates.append((priority, offset, index))
    return min(candidates)[2] if candidates else None


def _first_open_review_index(papers: list[dict[str, object]]) -> int:
    return _next_unread(papers) or 0


#: Heading of the review decision panel.  The panel answers what the queue
#: proposed, so the heading names the judgement rather than the whole task.
REVIEW_PANEL_TITLE = "自动判断"


def _render_review_workspace(
    database: Database,
    settings: Settings,
    snapshot: dict[str, object],
    profile: ProductProfile | None = None,
) -> None:
    """Laptop-first review workspace: reading stays beside the decision panel.

    One paper fills one screen: navigation and progress are a single compact
    row, the left column is the paper itself, and the right column is the AI's
    reasoning plus the human decision.  Provider internals and the raw judgment
    JSON live behind disclosures at the bottom of that column.
    """
    run_id = int(snapshot["run"]["id"])
    papers = snapshot["papers"]
    if not papers:
        st.info("暂无待审核论文。")
        return
    index_key = f"local_review_index_{run_id}"
    # Open on the work that is actually left.  Starting at paper 1 in a batch
    # that is already 99% decided puts an already-reviewed paper on screen while
    # the counter says "待审核 1", which is exactly the confusing state the
    # review page exists to avoid.  An explicit choice (previous/next, or a jump
    # from the verification list) is remembered and wins.
    saved_index = st.session_state.get(index_key)
    index = (
        int(saved_index)
        if saved_index is not None
        else _first_open_review_index(papers)
    )
    target_key = f"local_review_target_{run_id}"
    target_paper_id = st.session_state.pop(target_key, None)
    if target_paper_id is not None:
        for candidate_index, candidate in enumerate(papers):
            if int(candidate["id"]) == int(target_paper_id):
                index = candidate_index
                break
    index = max(0, min(index, len(papers) - 1))
    notice_key = f"local_review_notice_{run_id}"
    if notice := st.session_state.pop(notice_key, None):
        st.toast(str(notice))
    # Progress and navigation share one compact row: on a laptop the queue
    # position has to cost almost nothing, because the paper and the decision
    # panel are what the researcher is looking at.
    hosted = getattr(settings, "hosted", False)
    reviewed = sum(1 for paper in papers if paper.get("human_decision") != "undecided")
    with st.container(key="edr_review_nav"):
        progress_col, prev_col, next_col = st.columns(
            [2.4, 1, 1], vertical_alignment="center"
        )
        with progress_col:
            st.progress(
                reviewed / len(papers),
                text=f"已给决定 {reviewed}/{len(papers)} · 当前第 {index + 1} 篇",
            )
        if prev_col.button(
            "上一篇",
            disabled=index == 0,
            key=f"local_review_prev_{run_id}",
            use_container_width=True,
        ):
            st.session_state[f"remember_reader_{run_id}"] = True
            st.session_state[index_key] = max(0, index - 1)
            st.rerun()
        if next_col.button(
            "下一篇",
            disabled=index >= len(papers) - 1,
            key=f"local_review_next_{run_id}",
            use_container_width=True,
        ):
            st.session_state[f"remember_reader_{run_id}"] = True
            st.session_state[index_key] = min(len(papers) - 1, index + 1)
            st.rerun()

    paper = papers[index]
    paper_id = int(paper["id"])
    if profile and profile_store.profile_schema_ready(database):
        desk = ReadingDesk(database)
        if target_paper_id is not None or st.session_state.pop(f"remember_reader_{run_id}", False):
            desk.put("resume/" + profile.profile_id,
                     {"paper": paper_id, "identity": identity(paper), "run": run_id})
        saved = desk.shelf(profile.profile_id).get(paper_id) == identity(paper)
        if st.button("移出待读书架" if saved else "加入待读书架", key=f"reader_shelf_{paper_id}"):
            desk.put("resume/" + profile.profile_id,
                    {"paper": paper_id, "identity": identity(paper), "run": run_id})
            desk.bookmark(profile.profile_id, paper, not saved)
            st.rerun()
    processing = dict(paper.get("processing") or {})
    classification = dict(processing.get("classification") or {})
    chinese_title = str(paper.get("chinese_title") or paper["title"])
    english_title = str(paper["title"])
    doi = str(paper.get("doi") or "").strip()
    doi_url = build_doi_url(doi)
    access = resolve_paper_access(
        doi=doi,
        open_access_url=str(paper.get("open_access_url") or "") or None,
        source_url=str(paper.get("source_url") or "") or None,
        source_metadata=dict(paper.get("source_metadata") or {}),
    )
    doi_html = (
        '<a class="edr-doi" target="_blank" rel="noopener noreferrer" '
        f'href="{escape(doi_url, quote=True)}">{escape(doi)}</a>'
        if doi_url
        else '<span class="edr-doi">DOI 不可用</span>'
    )
    access_html = ""
    if access.preferred_url:
        access_html = (
            ' · <a class="edr-access-inline" target="_blank" '
            'rel="noopener noreferrer" '
            f'href="{escape(access.preferred_url, quote=True)}">'
            f"{escape(access.preferred_label or '公开页面')} ↗</a>"
        )
    meta = f"{paper.get('journal') or ''} · {str(paper.get('publication_date') or '')[:4]}"
    card = (
        '<div class="edr-paper-card">'
        f'<h3 class="edr-paper-title">{escape(chinese_title)}</h3>'
        f'<p class="edr-paper-title-en">{escape(english_title)}</p>'
        f'<p class="edr-paper-meta">{escape(meta)} · {doi_html}{access_html}</p>'
    )
    abstract = str(paper.get("abstract") or "")
    chinese_abstract = str(paper.get("chinese_abstract") or "")
    if abstract:
        card += f'<p class="edr-abstract">{escape(chinese_abstract or abstract)}</p>'
        source = zh_abstract_source(paper.get("abstract_source"))
        abstract_label = "中文译文" if chinese_abstract else "原始摘要"
        card += f'<p class="edr-quiet">{abstract_label} · 来源：{escape(source)}</p>' 
    elif hosted:
        card += '<p class="edr-quiet">此条目收录了题名与书目信息。</p>'
    else:
        retrieval = processing.get("abstract_retrieval")
        retrieval = retrieval if isinstance(retrieval, dict) else {}
        reason = zh_stop_reason(
            retrieval.get("final_reason")
            or (paper.get("abstract_enrichment") or {}).get("final_reason")
        )
        card += (
            '<p class="edr-quiet">暂未自动获取摘要 · '
            f"{escape(str(reason or '数据源未提供摘要'))}</p>"
        )
    card += "</div>"
    decision_key = f"local_review_{run_id}_{paper_id}_decision"
    note_key = f"local_review_{run_id}_{paper_id}_note"
    if decision_key not in st.session_state:
        human_decision = paper.get("human_decision") or "undecided"
        st.session_state[decision_key] = zh_human_decision(human_decision)
    if note_key not in st.session_state:
        st.session_state[note_key] = str(paper.get("reviewer_notes") or "")
    reading_col, review_col = st.columns([1.72, 1], gap="large")
    with reading_col, st.container(key="edr_review_reading_panel"):
        st.markdown(card, unsafe_allow_html=True)
        if abstract and chinese_abstract:
            excerpt = paper.get("abstract_source") == "publisher_excerpt"
            label = "查看英文节选" if excerpt else "查看英文原文"
            with st.expander(label):
                st.markdown(
                    f'<div class="edr-reading-text edr-reading-text--source">'
                    f"{escape(abstract)}</div>",
                    unsafe_allow_html=True,
                )
        elif not abstract and not hosted:
            st.info("暂未自动获取摘要")
            if st.button(
                "重新获取",
                key=f"local_review_refetch_{run_id}_{paper_id}",
            ):
                try:
                    with st.spinner("正在重试摘要数据源……"):
                        stats = _build_enrichment_service(database, settings).enrich_papers(
                            (paper_id,), workflow_batch_id=None, force_refresh=True,
                        )
                except Exception as exc:
                    _render_operation_error(
                        "摘要重新获取失败",
                        exc,
                        stage="abstract_retrieval_retry",
                        paper_id=paper_id,
                    )
                else:
                    st.session_state[notice_key] = (
                        "已获得原始摘要" if stats.enriched or stats.skipped_have_abstract
                        else "本次未获得公开摘要，已保留检索记录"
                    )
                    st.rerun()
            pasted = st.text_area(
                "粘贴英文摘要",
                key=f"local_review_paste_{run_id}_{paper_id}",
                height=120,
            )
            save_col, process_col = st.columns(2)
            if save_col.button(
                "只保存",
                key=f"local_review_paste_save_{run_id}_{paper_id}",
                use_container_width=True,
            ):
                try:
                    from eduecon_radar.services.manual_abstract import (
                        save_manual_abstract,
                    )

                    result = save_manual_abstract(database, paper_id, pasted, None)
                except Exception as exc:
                    _render_operation_error(
                        "英文摘要保存失败",
                        exc,
                        stage="manual_abstract_save",
                        paper_id=paper_id,
                    )
                else:
                    if result.get("error"):
                        st.error(str(result["error"]))
                    else:
                        st.toast("英文摘要已保存")
                        st.rerun()
            if process_col.button(
                "保存并分析",
                key=f"local_review_paste_process_{run_id}_{paper_id}",
                disabled=not getattr(settings, "deepseek_api_key", None),
                use_container_width=True,
            ):
                try:
                    from eduecon_radar.services.manual_abstract import (
                        save_manual_abstract,
                    )

                    saved = save_manual_abstract(database, paper_id, pasted, None)
                    if saved.get("error"):
                        st.error(str(saved["error"]))
                    else:
                        with st.spinner("正在翻译并初筛这一篇……"):
                            process_batch(
                                database,
                                run_id,
                                (paper_id,),
                                _build_services(database, settings, profile),
                            )
                except Exception as exc:
                    _render_operation_error(
                        "保存并分析失败",
                        exc,
                        stage="manual_abstract_processing",
                        paper_id=paper_id,
                    )
                else:
                    st.toast("已保存并完成分析")
                    st.rerun()

    with review_col, st.container(key="edr_review_decision_panel"):
        # A review screen is a decision tool, not a report about the model.  The
        # everyday view answers three things and nothing else: what the queue
        # proposed, how much evidence that proposal had, and what you decide.
        # The reasoning, quotes and condition fields remain available behind the
        # disclosure below for the times they are actually being audited.
        decision_zh = zh_auto_decision(classification.get("decision"))
        judgment_label = screening_state_label(
            screening_state_from_classification(classification)
        )
        decision_badge = status_badge_html(
            decision_zh, _auto_decision_tone(classification.get("decision"))
        )
        st.markdown(
            '<div class="edr-decision-head">'
            f'<div class="edr-review-panel-title">{REVIEW_PANEL_TITLE}</div>'
            f'<div class="edr-decision-badges">{decision_badge}'
            f'<span class="edr-decision-basis">{escape(judgment_label)}</span></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        human_decision = paper.get("human_decision") or "undecided"
        suggestion = None
        if human_decision == "undecided":
            suggestion = {
                "relevant": "相关",
                "irrelevant": "无关",
                "uncertain": "不确定",
            }.get(str(classification.get("decision") or ""))
        # The suggestion has to be read *before* the choice, not after it, so the
        # caption sits above the control it is advising on.
        if suggestion:
            st.markdown(
                f'<p class="edr-decision-hint">AI 建议：{escape(suggestion)}</p>',
                unsafe_allow_html=True,
            )
        elif human_decision != "undecided":
            st.markdown(
                '<p class="edr-decision-hint edr-decision-hint--quiet">'
                "这篇论文已有你保存的人工决定，修改后再次保存即可覆盖。</p>",
                unsafe_allow_html=True,
            )
        notes = paper.get("source_metadata", {}).get("reading_notes", {})
        if profile and (reading_note := notes.get(profile.profile_id)):
            st.caption("阅读线索 · 编者整理")
            st.write(reading_note)
        reason = str(classification.get("reason") or "").strip()
        if reason:
            preview = reason if len(reason) <= 180 else reason[:177].rstrip() + "…"
            st.markdown(
                f'<p class="edr-decision-hint">AI 理由 · {escape(preview)}</p>',
                unsafe_allow_html=True,
            )
        with st.form(f"review_decision_{run_id}_{paper_id}", enter_to_submit=False):
            decision = st.radio(
                "人工决定",
                list(HUMAN_DECISION_OPTIONS),
                horizontal=True,
                key=decision_key,
                label_visibility="collapsed",
            )
            # The note belongs to the decision and is saved by the same button, so it
            # is read before the draft is assembled.
            with st.expander("添加个人备注（可选）"):
                note = st.text_input(
                    "备注内容",
                    key=note_key,
                    placeholder="只记录需要回看的信息",
                    max_chars=1000 if hosted else None,
                    label_visibility="collapsed",
                )
            drafts = {paper_id: (human_decision_from_zh(decision), note)}
            # One save path, not two: "save and continue" also stores the decision on
            # the last paper, so there is no unreachable state at the end of a queue
            # and no second button that does almost the same thing.
            if st.form_submit_button(
                "保存并下一篇",
                type="primary",
                key=f"local_review_save_next_{run_id}",
                use_container_width=True,
            ):
                try:
                    save_human_decisions(database, run_id, drafts)
                except Exception as exc:
                    _render_operation_error("人工决定保存失败", exc, stage="human_review")
                else:
                    st.session_state.pop("library_export_data", None)
                    st.session_state.pop(decision_key, None)
                    st.session_state.pop(note_key, None)
                    st.session_state[f"remember_reader_{run_id}"] = True
                    following = _next_unread(papers, index)
                    st.session_state[index_key] = following if following is not None else index
                    st.session_state[notice_key] = (
                        "已保存，进入下一篇待审核论文" if following is not None else "已保存"
                    )
                    st.rerun()
        with st.expander("AI 判断依据与原文证据"):
            st.markdown(
                _ai_reasoning_block(
                    classification,
                    list(classification.get("evidence") or []),
                    has_abstract=bool(abstract), title=english_title, abstract=abstract,
                ),
                unsafe_allow_html=True,
            )
            retrieval = processing.get("abstract_retrieval")
            if retrieval:
                st.write("摘要获取记录")
                st.json(retrieval)
            if classification:
                st.write("自动判断记录")
                st.json(classification)
            else:
                st.write("暂无自动判断详情")
        if not hosted and paper.get("abstract"):
            with st.expander("更多操作"):
                if st.button(
                    "重新翻译本文摘要",
                    key=f"local_review_retranslate_{run_id}_{paper_id}",
                    use_container_width=True,
                ):
                    try:
                        with st.spinner("正在重新翻译中文摘要……"):
                            retranslate_abstract(
                                database,
                                run_id,
                                paper_id,
                                _build_services(database, settings, profile),
                            )
                    except Exception as exc:
                        _render_operation_error(
                            "重新翻译摘要失败",
                            exc,
                            stage="abstract_retranslation",
                            paper_id=paper_id,
                        )
                    else:
                        st.rerun()
                if st.button(
                    "重新筛选本文",
                    key=f"local_review_rescreen_{run_id}_{paper_id}",
                    use_container_width=True,
                ):
                    try:
                        with st.spinner("正在重新筛选本文……"):
                            rescreen_paper(
                                database,
                                run_id,
                                paper_id,
                                _build_services(database, settings, profile),
                            )
                    except Exception as exc:
                        _render_operation_error(
                            "重新筛选失败",
                            exc,
                            stage="rescreen",
                            paper_id=paper_id,
                        )
                    else:
                        st.rerun()


def _render_verification_summary_section(
    database: Database,
    settings: Settings,
    registry: JournalRegistry,
    snapshot: dict[str, object],
    profile: ProductProfile,
) -> None:
    """Completeness audit: one status line, the panel behind a disclosure.

    Answering "did the retrieval miss anything?" costs an independent-provider
    query and a full comparison table.  That is a periodic check, not a step in
    the daily read-review-export loop, so it must not occupy the position where
    the next action belongs.
    """
    run = snapshot["run"]
    run_id = int(run["id"])
    try:
        result = cached_verification(database, run_id)
    except ProductSafetyMigrationRequired:
        return
    except Exception as exc:
        LOGGER.exception("完整性核验状态读取失败 run_id=%s", run_id)
        st.caption("完整性核验状态暂时无法读取；论文获取、审核与导出不受影响。")
        if st.toggle("查看核验错误详情", key=f"verification_error_{run_id}"):
            st.code(_safe_error(exc), language=None)
        return
    labels = {
        "consistent": "一致",
        "broadly_consistent": "基本一致",
        "possible_gaps": "发现可能遗漏",
        "unavailable": "暂无法核验",
    }
    tones: dict[str, StatusTone] = {
        "consistent": "complete",
        "broadly_consistent": "info",
        "possible_gaps": "review",
        "unavailable": "missing",
    }
    if result is None:
        status_text = "尚未与独立来源对照"
        tone: StatusTone = "pending"
    else:
        status_text = (
            f"{labels[result.status]} · 主来源 {result.local_count} 篇 · "
            f"疑似遗漏 {result.possible_missing_count}"
        )
        tone = tones[result.status]
    st.markdown(
        '<div class="edr-audit-line">'
        '<span class="edr-audit-label">检索完整性</span>'
        f"{status_badge_html(status_text, tone)}"
        '<span class="edr-audit-note">独立来源对照，只检查是否漏项，不改变论文列表</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander(
        "查看完整性核验",
        expanded=bool(result is not None and result.status == "possible_gaps"),
    ):
        _render_retrieval_verification(database, settings, registry, snapshot, profile)


def _render_acquired_detail(snapshot: dict[str, object]) -> None:
    """The acquisition record: what was asked for, and what came back.

    Folded shut by default.  Provider page counts and stop reasons explain a
    result that looks wrong; they are not part of reading a paper.
    """
    run = snapshot["run"]
    journal_keys = json.loads(run.get("selected_journal_keys") or "[]")
    matched_ids = json.loads(run.get("matched_paper_ids") or "[]")
    if not st.toggle("查看本次获取明细", key=f"run_details_{run['id']}"):
        return
    st.dataframe(
        [
            {
                "期刊数": len(journal_keys),
                "找到": run.get("provider_returned_count"),
                "去重后": len(matched_ids),
                "停止原因": zh_stop_reason(run.get("stopped_reason")),
            }
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.dataframe(
        [
            {
                "中文标题": paper.get("chinese_title") or "",
                "英文标题": paper["title"],
                "期刊": paper.get("journal") or "",
                "年份": str(paper.get("publication_date") or "")[:4],
                "DOI": build_doi_url(paper.get("doi")) or "",
                "状态": status_label(str(paper["processing"].get("status") or "pending")),
                "摘要来源": zh_abstract_source(paper.get("abstract_source")),
            }
            for paper in snapshot["papers"]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={"DOI": st.column_config.LinkColumn("DOI", display_text="打开 DOI")},
    )


def _verification_summary(result: VerificationResult) -> str:
    if result.status == "consistent":
        return "本次结果与独立来源逐条吻合，未发现明显漏项。"
    if result.status == "broadly_consistent":
        return "两边结果总体接近；少量差异可由题名或收录口径解释，未发现明确漏项。"
    if result.status == "possible_gaps":
        return (
            f"独立来源中有 {result.possible_missing_count} 篇可比记录未在本次结果中找到，"
            "建议查看下方论文并人工确认。"
        )
    return (
        f"独立来源暂时无法连接，目前还不能判断是否漏项；"
        f"已抓到的 {result.local_count} 篇论文不受影响。"
    )


def _confirmed_overlap(result: VerificationResult) -> int | None:
    if result.status == "unavailable":
        return None
    stored = result.details.get("confirmed_overlap_count")
    if isinstance(stored, int) and not isinstance(stored, bool):
        return stored
    if result.comparable_independent_count is None:
        return None
    # Old cached rows predate the explicit metric.  The independent comparable
    # set is partitioned into confirmed records and possible gaps.
    return max(0, result.comparable_independent_count - result.possible_missing_count)


def _intake_badge(state: CandidateIntakeState) -> tuple[str, str]:
    if state.tombstoned:
        return "曾被移除 · 需确认恢复", "failed"
    if not state.in_library:
        return "可加入", "review"
    if not state.attached_to_run:
        return "已在论文库 · 未接入本批", "info"
    if state.human_decision not in {None, "undecided"}:
        return "已审核", "complete"
    if state.review_ready:
        return "已进入审核", "complete"
    if state.processing_status in {"needs_retry", "failed", "needs_manual_abstract"}:
        return "已加入 · 需要处理", "retry"
    return "已加入 · 处理中", "pending"


def _candidate_key(run_id: int, item: dict[str, object], index: int) -> str:
    doi = str(item.get("doi") or "").strip()
    return f"verify_intake_{run_id}_{doi or f'idx{index}'}"


def _render_candidate_actions(
    database: Database,
    settings: Settings,
    registry: JournalRegistry,
    run_id: int,
    item: dict[str, object],
    state: CandidateIntakeState,
    index: int,
) -> None:
    key = _candidate_key(run_id, item, index)
    badge_label, badge_tone = _intake_badge(state)
    render_status_badge(badge_label, badge_tone)  # type: ignore[arg-type]
    if state.tombstoned:
        warning = "这篇论文此前被你主动移除并归档。"
        if state.tombstone_has_human_data:
            warning += " 归档中包含人工决定或编辑，恢复后会重新进入处理链。"
        st.warning(warning)
        confirmed = st.checkbox(
            "我确认要恢复这篇论文并重新加入处理链",
            key=f"{key}_restore_confirm",
        )
        if st.button(
            "恢复并加入处理",
            type="primary",
            disabled=not confirmed,
            key=f"{key}_restore",
            use_container_width=True,
        ):
            _run_candidate_intake(
                database,
                settings,
                registry,
                run_id,
                (item,),
                allow_restore=True,
                label="恢复并加入处理",
            )
        return
    if not state.in_library or not state.attached_to_run:
        label = (
            "加入并处理"
            if not state.in_library
            else "接入本批处理"
        )
        if st.button(
            label,
            type="primary",
            key=f"{key}_intake",
            use_container_width=True,
        ):
            _run_candidate_intake(
                database,
                settings,
                registry,
                run_id,
                (item,),
                label=label,
            )
        return
    if state.review_ready:
        if st.button(
            "去审核",
            type="primary",
            key=f"{key}_review",
            use_container_width=True,
        ):
            st.session_state[f"local_review_target_{run_id}"] = int(state.paper_id)
            st.rerun()
        return
    if state.human_decision not in {None, "undecided"}:
        return
    if st.button(
        "继续处理",
        key=f"{key}_reprocess",
        use_container_width=True,
    ):
        _run_candidate_intake(
            database,
            settings,
            registry,
            run_id,
            (item,),
            label="继续处理",
        )


def _intake_metadata_clients(settings: Settings) -> tuple[object, object]:
    from eduecon_radar.services.crossref import CrossrefClient
    from eduecon_radar.services.openalex import OpenAlexClient

    return CrossrefClient(), OpenAlexClient(settings.openalex_api_key)


def _run_candidate_intake(
    database: Database,
    settings: Settings,
    registry: JournalRegistry,
    run_id: int,
    candidates: tuple[dict[str, object], ...],
    *,
    allow_restore: bool = False,
    label: str,
    profile: ProductProfile | None = None,
) -> None:
    """User-triggered intake; the only place providers/AI may run."""
    try:
        if profile is None:
            registry_config = load_profile_registry(
                settings.profiles_root, catalog_path=settings.journal_catalog_path,
                default_profile_id=settings.default_profile,
            )
            profile = registry_config.profile(run_profile_id(database, run_id))
        services = _build_services(database, settings, profile)
        crossref, openalex = _intake_metadata_clients(settings)
    except Exception as exc:
        _render_operation_error("处理服务初始化失败", exc, stage="intake_services")
        return
    try:
        with st.status(f"正在{label}", expanded=True) as progress:
            results = intake_verification_candidates(
                database,
                run_id,
                candidates,
                registry=registry,
                crossref=crossref,
                openalex=openalex,
                services=services,
                allow_restore=allow_restore,
                progress=lambda text: progress.update(label=text),
            )
            entered = [result for result in results if result.entered_chain]
            already = [
                result
                for result in results
                if result.outcome == "already_in_library"
            ]
            ready = sum(1 for result in results if result.review_ready)
            skipped = [result for result in results if result.outcome == "skipped"]
            errors = [
                message
                for result in results
                for message in result.errors
                if message
            ]
            summary_parts = []
            if entered:
                summary_parts.append(f"已加入 {len(entered)} 篇")
            if already:
                summary_parts.append(f"已在论文库 {len(already)} 篇")
            if ready:
                summary_parts.append(f"已进入审核 {ready} 篇")
            if skipped:
                summary_parts.append(f"跳过 {len(skipped)} 篇")
            if errors:
                summary_parts.append(f"待处理 {len(errors)} 项")
            progress.update(
                label=("；".join(summary_parts) or "没有新的可加入论文") + "。",
                state="complete",
            )
    except Exception as exc:
        _render_operation_error(
            "候选加入处理失败；已完成的论文保留，可逐篇重试",
            exc,
            stage="intake_run",
        )
        return
    if entered:
        st.toast(f"已加入 {len(entered)} 篇论文并接入处理链")
    elif already:
        st.toast(f"{len(already)} 篇论文已在论文库中，无需重复加入")
    if skipped:
        message = "；".join(
            f"{result.title or '未知论文'}：{'；'.join(result.errors)}"
            for result in skipped
        )
        st.session_state[f"local_intake_notice_{run_id}"] = ("error", message)
    elif errors:
        st.session_state[f"local_intake_notice_{run_id}"] = (
            "warning",
            "；".join(dict.fromkeys(errors)),
        )
    if ready:
        st.session_state[f"local_review_notice_{run_id}"] = (
            f"本次核验加入的 {ready} 篇论文已进入人工审核。"
        )
    st.rerun()


def _render_possible_missing(
    database: Database,
    settings: Settings,
    registry: JournalRegistry,
    run_id: int,
    result: VerificationResult,
) -> None:
    possible = list(result.details.get("possible_missing") or [])
    if not possible:
        return
    st.markdown(
        '<div class="edr-verification-list-title">需要人工确认的论文</div>',
        unsafe_allow_html=True,
    )
    states = list(candidate_intake_states(database, run_id, possible))
    actionable = [
        (item, state)
        for item, state in zip(possible, states, strict=True)
        if not state.tombstoned and (not state.in_library or not state.attached_to_run)
    ]
    if len(actionable) > 1:
        if st.button(
            f"全部加入并处理（{len(actionable)} 篇）",
            type="primary",
            key=f"verify_intake_all_{run_id}",
            use_container_width=True,
        ):
            _run_candidate_intake(
                database,
                settings,
                registry,
                run_id,
                tuple(item for item, _state in actionable),
                label="加入全部候选论文",
            )
        st.caption(
            "点击后逐篇执行：加入文献库 → 中文标题 → 英文摘要 → 中文摘要 → 自动初筛。"
            "任何一篇失败都不影响其他候选。"
        )
    for index, (item, state) in enumerate(zip(possible, states, strict=True)):
        title = escape(str(item.get("title") or "题名暂缺"))
        doi = str(item.get("doi") or "").strip()
        source_id = str(item.get("source_id") or "").strip()
        access = resolve_paper_access(
            doi=doi,
            openalex_id=source_id,
        )
        links = []
        if access.doi_url:
            links.append(
                f'<a href="{escape(access.doi_url, quote=True)}" target="_blank" '
                'rel="noopener noreferrer">DOI ↗</a>'
            )
        if access.preferred_url:
            links.append(
                f'<a href="{escape(access.preferred_url, quote=True)}" '
                'target="_blank" rel="noopener noreferrer">'
                f"{escape(access.preferred_label or '公开页面')} ↗</a>"
            )
        link_html = " · ".join(links) or "<span>暂无外部链接</span>"
        publication_date = escape(str(item.get("publication_date") or "日期未知"))
        doi_text = escape(doi) if doi else "DOI 暂缺"
        reason = escape(str(item.get("reason") or "独立来源存在可比记录，主结果中未找到匹配项"))
        crossref_dates = dict(item.get("crossref_dates") or {})
        date_compare = ""
        if crossref_dates:
            created = escape(str(crossref_dates.get("created") or "未知"))
            publication_basis = escape(
                str(
                    crossref_dates.get("published_online")
                    or crossref_dates.get("published_print")
                    or crossref_dates.get("issued")
                    or "未知"
                )
            )
            date_compare = (
                '<div class="edr-verification-date-compare">'
                f"OpenAlex {publication_date} · Crossref 创建 {created} · "
                f"Crossref 出版口径 {publication_basis}</div>"
            )
        st.markdown(
            "".join(
                (
                    '<div class="edr-verification-missing">',
                    f'<div class="edr-verification-missing-title">{title}</div>',
                    '<div class="edr-verification-missing-meta">',
                    f"{publication_date} · {doi_text} · {link_html}",
                    "</div>",
                    date_compare,
                    f'<div class="edr-verification-reason">{reason}</div>',
                    "</div>",
                )
            ),
            unsafe_allow_html=True,
        )
        _render_candidate_actions(
            database,
            settings,
            registry,
            run_id,
            item,
            state,
            index,
        )


def _render_verification_technical_details(result: VerificationResult) -> None:
    st.dataframe(
        [
            {
                "主来源记录": result.local_count,
                "独立来源全部记录": result.independent_count,
                "独立来源可比记录": result.comparable_independent_count,
                "相同 DOI": result.exact_doi_matches,
                "来源侧未匹配": result.source_only_count,
                "本地侧未匹配": result.local_only_count,
            }
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(str(result.details.get("date_basis") or "暂无日期口径说明"))
    st.caption(
        f"来源：{result.source_name} · 用时 {result.duration_ms / 1000:.1f} 秒 · "
        f"网络请求 {result.query_count} 次"
    )
    if result.error_message:
        st.code(result.error_message, language=None)
    detail_groups = (
        ("题名匹配", result.details.get("title_matches")),
        ("已解释的来源差异", result.details.get("explained_source_only")),
        ("本地主来源独有", result.details.get("local_only")),
    )
    for label, rows in detail_groups:
        values = list(rows or [])
        if values:
            st.markdown(f"**{label}**")
            st.dataframe(values, hide_index=True, use_container_width=True)


def _render_retrieval_verification(
    database: Database,
    settings: Settings,
    registry: JournalRegistry,
    snapshot: dict[str, object],
    profile: ProductProfile | None = None,
) -> None:
    run_id = int(snapshot["run"]["id"])
    try:
        result = cached_verification(database, run_id)
    except ProductSafetyMigrationRequired:
        return
    except Exception as exc:
        st.warning("完整性核验结果暂时无法读取；本次论文结果不受影响。")
        if st.toggle("技术详情", key=f"verification_technical_error_{run_id}"):
            st.code(_safe_error(exc), language=None)
        return
    journal_keys = json.loads(snapshot["run"].get("selected_journal_keys") or "[]")
    try:
        journal_issns = tuple(registry.by_key(key).issns for key in journal_keys)
    except KeyError:
        journal_issns = ()
    with st.container(key="edr_verification_result"):
        notice_key = f"local_intake_notice_{run_id}"
        notice = st.session_state.pop(notice_key, None)
        if notice is not None:
            kind, message = notice
            (st.error if kind == "error" else st.warning)(str(message))
        st.markdown(
            '<div class="edr-verification-title">完整性核验结果</div>',
            unsafe_allow_html=True,
        )
        if result is None:
            st.markdown(
                '<div class="edr-verification-empty">'
                "尚未与独立来源对照。核验只检查是否存在明显漏项，"
                "不会改变当前论文列表。</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="edr-verification-summary">'
                f"{escape(_verification_summary(result))}</div>",
                unsafe_allow_html=True,
            )
            overlap = _confirmed_overlap(result)
            independent = result.comparable_independent_count
            metric_values = (
                (result.local_count, "主来源论文"),
                ("—" if independent is None else independent, "独立来源可比"),
                ("—" if overlap is None else overlap, "已确认重合"),
                (
                    "—" if result.status == "unavailable" else result.possible_missing_count,
                    "疑似遗漏",
                ),
            )
            metric_html = "".join(
                f"<div><b>{escape(str(value))}</b><span>{label}</span></div>"
                for value, label in metric_values
            )
            st.markdown(
                f'<div class="edr-verification-stats">{metric_html}</div>',
                unsafe_allow_html=True,
            )
            if result.status == "possible_gaps":
                _render_possible_missing(database, settings, registry, run_id, result)
        if st.button(
            "重新核验" if result else "开始核验",
            key=f"verify_retrieval_{run_id}",
            disabled=not journal_issns,
        ):
            from eduecon_radar.services.crossref import CrossrefClient
            from eduecon_radar.services.retrieval_verification import (
                OpenAlexVerificationSource,
                verify_sync_run,
            )

            with st.status("正在核验完整性", expanded=True) as progress:
                progress.update(label="正在查询独立来源（OpenAlex）…")
                verification_result = verify_sync_run(
                    database,
                    run_id,
                    journal_issns=journal_issns,
                    source=OpenAlexVerificationSource(settings.openalex_api_key),
                    crossref=CrossrefClient(),
                )
                if verification_result.status == "possible_gaps":
                    progress.update(
                        label=(
                            f"核验完成：发现 {verification_result.possible_missing_count} 篇"
                            "疑似遗漏，已列出待人工确认。"
                        ),
                        state="complete",
                    )
                else:
                    progress.update(
                        label="核验完成，结果已更新。",
                        state="complete",
                    )
            st.rerun()
        if result is not None and st.toggle("技术详情", key=f"verification_technical_{run_id}"):
            _render_verification_technical_details(result)


def _render_exports(database: Database, snapshot: dict[str, object], *, allow_archive=True) -> None:
    """Exports of this batch, named by what the file is for.

    The previous version asked the reader to choose a "導師版/研究版" over a
    raw scope concept and read a seven-file dump path.  What a researcher
    actually picks between is *what the file is for*: a reading list, their own
    record, or something safe to hand to someone else.
    """
    render_section_header("导出")
    with st.container(key="edr_export_panel"):
        _render_export_controls(database, snapshot)
        if allow_archive:
            _render_batch_export_pack(database, snapshot)


def _render_batch_export_pack(database: Database, snapshot: dict[str, object]) -> None:
    """The full multi-file pack, kept for reproducibility and clearly labelled."""
    run_id = int(snapshot["run"]["id"])
    papers = snapshot["papers"]
    with st.expander("生成完整成果包（7 个文件）"):
        st.markdown(
            '<p class="edr-action-hint">一次性写出本批论文的完整成果文件，'
            "包含逐篇记录与核对表；适合归档或复核，不需要逐项选择。</p>",
            unsafe_allow_html=True,
        )
        if st.button("生成成果包", key=f"local_overnight_exports_{run_id}"):
            try:
                from eduecon_radar.services.overnight_exports import (
                    write_overnight_exports,
                )

                paths = write_overnight_exports(
                    database,
                    tuple(int(paper["id"]) for paper in papers),
                    sync_run_id=run_id,
                )
            except Exception as exc:
                _render_operation_error("成果导出失败", exc, stage="overnight_exports")
            else:
                st.success(
                    "已生成 7 个成果文件："
                    + "；".join(str(path) for path in vars(paths).values())
                )


def _render_export_controls(database: Database, snapshot: dict[str, object]) -> None:
    run_id = int(snapshot["run"]["id"])
    papers = snapshot["papers"]
    st.markdown(
        '<p class="edr-export-intro">导出本批论文。每份文件都可以下载 CSV（表格）'
        "或 Markdown（纯文本）。</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="edr-export-choice"><b>导师分享版</b>'
        "<span>只含有依据的内容：原始文献字段、正式翻译与独立人工备注，"
        "不含 AI 判断与个人工作信息。</span></div>",
        unsafe_allow_html=True,
    )
    mentor_csv = export_mentor_csv(papers)
    mentor_markdown = export_mentor_markdown(papers)
    mentor_left, mentor_right = st.columns(2)
    mentor_left.download_button(
        "下载 CSV（表格）",
        mentor_csv,
        file_name="literature_radar_mentor.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"export_mentor_csv_{run_id}",
    )
    mentor_right.download_button(
        "下载 Markdown（纯文本）",
        mentor_markdown,
        file_name="literature_radar_mentor.md",
        mime="text/markdown",
        use_container_width=True,
        key=f"export_mentor_md_{run_id}",
    )
    st.markdown(
        '<div class="edr-export-choice"><b>研究数据库</b>'
        "<span>完整记录：自动判断、判断依据、处理状态与个人决定，"
        "适合长期归档与后续分析。</span></div>",
        unsafe_allow_html=True,
    )
    research_csv = export_research_csv(papers)
    research_markdown = export_research_markdown(papers)
    research_left, research_right = st.columns(2)
    research_left.download_button(
        "下载 CSV（表格）",
        research_csv,
        file_name="literature_radar_research.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"export_research_csv_{run_id}",
    )
    research_right.download_button(
        "下载 Markdown（纯文本）",
        research_markdown,
        file_name="literature_radar_research.md",
        mime="text/markdown",
        use_container_width=True,
        key=f"export_research_md_{run_id}",
    )


def _build_enrichment_service(database: Database, settings: Settings) -> AbstractEnrichmentService:
    from eduecon_radar.services.pipeline_builder import build_enrichment_service

    return build_enrichment_service(database, settings)


def _build_batch_title_service(
    database: Database, settings: Settings
) -> BatchTitleTranslationService:
    from eduecon_radar.services.pipeline_builder import build_batch_title_service

    return build_batch_title_service(database, settings)


def _build_translation_service(database: Database, settings: Settings) -> TranslationService:
    from eduecon_radar.services.pipeline_builder import build_translation_service

    return build_translation_service(database, settings)


def _build_services(
    database: Database,
    settings: Settings,
    profile: ProductProfile | None = None,
) -> LocalPipelineServices:
    from eduecon_radar.services.pipeline_builder import build_local_pipeline_services

    return build_local_pipeline_services(database, settings, profile=profile)


def _journal_label(journal: RegistryJournal) -> str:
    return f"{journal.canonical_name} ({journal.short_name}) · {journal.print_issn}" + (
        f" / {journal.online_issn}" if journal.online_issn else ""
    )


def _safe_error(exc: Exception) -> str:
    return str(exc).replace("Authorization", "[redacted]")[:500]


def _render_operation_error(
    message: str,
    exc: Exception,
    *,
    stage: str,
    paper_id: int | None = None,
) -> None:
    safe = _safe_error(exc)
    LOGGER.exception("%s at stage=%s paper_id=%s", message, stage, paper_id)
    st.error(f"{message}：{safe}")
