"""The daily paper-first view. Ranking uses local evidence only."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import streamlit as st

from eduecon_radar.services.reading_desk import (
    ReadingDesk,
    identity,
    parse_topics,
    to_ris,
    tracking_window,
    update_watch,
)
from eduecon_radar.services.ui_labels import zh_auto_decision, zh_human_decision
from eduecon_radar.ui.views import paper_library

FILTERS = dict(
    journal="全部",
    abstract_state="全部",
    decision="全部",
    auto_decision="全部",
    year="全部",
    search="",
)


def _view(name):
    st.session_state["radar_nav"] = name
    st.rerun()


def _watch_controls(database, queue, settings, profile, journals, defaults):
    desk = ReadingDesk(database)
    pid = profile.profile_id
    watch = desk.watch(pid)
    with st.expander("关注范围与更新"), st.form("watch_" + pid):
        selected = st.multiselect(
            "关注期刊",
            list(journals),
            default=[k for k in watch.get("journals", defaults) if k in journals],
            format_func=lambda k: journals[k].canonical_name,
        )
        topics = st.text_input(
            "兴趣词",
            value="，".join(watch.get("topics", [])),
            max_chars=240,
            help="用逗号分隔。匹配已有的中英文题名与摘要，可以同时填写中英文。",
        )
        auto = st.checkbox("打开本方向的雷达时自动更新", value=watch.get("automatic", False))
        st.caption(
            "首次查近 30 天。之后从上次查全的位置向前回看 7 天，补收迟到记录。"
            "追踪只查公开书目，不调用付费 AI。"
        )
        if st.form_submit_button("保存关注范围"):
            if selected:
                desk.save_watch(pid, selected, parse_topics(topics), auto, date.today())
                st.rerun()
            st.warning("请至少选择一本期刊。")
    if not watch:
        return []
    keys = watch["journals"]
    if any(k not in journals for k in keys):
        st.warning("关注列表中有期刊已移出目录，请重新保存关注范围。")
        return watch["topics"]
    selected = [journals[k] for k in keys]
    today = date.today()
    since, frontier, last = tracking_window(database, queue, pid, watch, today)
    coverage = f"已查至 {frontier}" if frontier else f"从 {since} 起追踪"
    status, button = st.columns([4, 1])
    status.caption(f"关注 {len(keys)} 本期刊 · {coverage}")
    manual = button.button("检查更新", key="check_" + pid)
    if watch["automatic"] or manual:
        job_id = update_watch(database, queue, pid, watch, selected, today,
                              retry_partial=manual, window=(since, frontier, last))
        if not last or job_id != last["id"]:
            queue.start(settings)
            st.rerun()
        if manual:
            st.toast("本次进度已保存" if last["phase"] != "done" else "今天的检索已完成")
    if last and last["phase"] == "done" and frontier != date.fromisoformat(last["until"]):
        st.caption("上次部分来源未返回完整结果，下次更新会再次覆盖这段日期。")
    return watch["topics"]


def render(database, settings, profile, queue, journals, defaults):
    hosted = getattr(settings, "hosted", False)
    desk = ReadingDesk(database)
    pid = profile.profile_id
    scope = paper_library.resolve_library_scope(database, pid)
    focus_key = "radar_focus_" + pid
    focused = st.session_state.get(focus_key)

    def open_paper(paper):
        st.session_state[focus_key] = {"paper": paper["id"], "identity": identity(paper)}
        desk.put("resume/" + pid, st.session_state[focus_key])
        st.rerun()

    if focused:
        papers = paper_library._library_papers(
            database, **FILTERS, scope=replace(scope, shared=True), paper_ids=(focused["paper"],)
        )
        if st.button("返回今日雷达") or not papers or identity(papers[0]) != focused["identity"]:
            st.session_state.pop(focus_key)
            st.rerun()
        paper_library._render_paper_card(database, settings, papers[0], profile, opened=True)
        return

    st.header("今日雷达")
    if hosted:
        topics = []
        st.caption("从公开期刊选入四篇论文。刷新或离开后，本次决定、备注和书架可能重置，可先导出。")
    else:
        topics = _watch_controls(database, queue, settings, profile, journals, defaults)
    resume = desk.get("resume/" + pid)
    if resume:
        last = paper_library._library_papers(
            database,
            **FILTERS,
            scope=replace(scope, shared=True),
            paper_ids=(resume["paper"],),
            summary_only=True,
        )
        if (
            last
            and identity(last[0]) == resume["identity"]
            and st.button(
                "接着读 · " + str(last[0].get("chinese_title") or last[0]["title"]),
                key="continue_" + pid,
            )
        ):
            if resume.get("run"):
                st.session_state["just_fetched_run"] = resume["run"]
                st.session_state[f"local_review_target_{resume['run']}"] = resume["paper"]
                _view("文献工作台")
            open_paper(last[0])
    section = (
        st.segmented_control(
            "阅读视图",
            ("优先阅读", "待读书架") if hosted
            else ("优先阅读", "近期发表", "本周新收", "待读书架"),
            default="优先阅读",
            key="radar_section_" + pid,
            label_visibility="collapsed",
        )
        or "优先阅读"
    )
    size_key = f"radar_size_{pid}_{section}"
    size = st.session_state.get(size_key, 5)
    revisiting = False
    if section == "待读书架":
        shelf = desk.shelf(pid)
        papers = (
            paper_library._library_papers(
                database, **FILTERS, scope=replace(scope, shared=True), paper_ids=tuple(shelf)
            )
            if shelf
            else []
        )
        records = {p["id"]: p for p in papers if identity(p) == shelf[p["id"]]}
        selected = [(records[k], []) for k in shelf if k in records]
        total = len(selected)
        if selected:
            st.download_button(
                f"导出书架 · {len(selected)} 篇 RIS",
                to_ris([p for p, _ in selected]),
                file_name=f"{pid}_reading.ris",
                mime="application/x-research-info-systems",
            )
        else:
            st.info("书架还是空的。在论文详情或逐篇审核时，点「加入待读书架」即可留到下次。")
    else:
        selected, total = paper_library.radar_page(
            database, scope, section, topics, size, date.today()
        )
        notes = {
            "近期发表": "近 30 天发表、仍待你判断的论文，按库中发表日期排列。",
            "本周新收": "近 7 天首次检索到、仍待你判断的论文，按本方向首次收录日期排列。",
            "优先阅读": "仅列待判断论文。先看兴趣词命中数，再参考本方向初筛与发表日期。",
        }
        if section == "优先阅读" and not total:
            filters = dict(FILTERS, decision="相关")
            total = paper_library._library_count(database, **filters, scope=scope, has_status=True)
            if total:
                revisiting = True
                saved = paper_library._library_papers(
                    database, **filters, scope=scope, limit=size, summary_only=True,
                )
                selected = [(paper, []) for paper in saved]
                st.caption(f"本方向的待审论文已处理完。你曾标为相关的 {total} 篇论文可随时重读。")
        if not revisiting:
            st.caption("按发表日期排列，点击题名阅读。" if hosted else notes[section])
        if not selected:
            st.info("这个范围内暂时没有待审论文。")
            if hosted and st.button("浏览全部方向的论文"):
                st.session_state["library_scope_picker"] = "全部方向"
                _view("论文库")
            if not hosted and st.button("前往文献工作台"):
                _view("文献工作台")
    for paper, hits in selected[:size]:
        with st.container(border=True, key=f"radar_card_{paper['id']}"):
            pub_date = str(paper.get("publication_date") or "")
            if pub_date > date.today().isoformat():
                pub_date = "预排刊期 " + pub_date
            st.caption(" · ".join(filter(None, [paper.get("journal"), pub_date])))
            title = paper_library._clean_title(paper.get("chinese_title") or paper["title"])
            if st.button(
                title,
                key=f"radar_open_{paper['id']}",
                type="tertiary",
                help="查看摘要、人工决定与阅读笔记",
            ):
                open_paper(paper)
            classification = paper["processing"]["classification"]
            basis = classification.get("screening_basis")
            evidence = {
                "title_only": "仅标题",
                "title_and_abstract": "标题与摘要",
                "deterministic_metadata_rule": "书目规则",
            }.get(basis, "依据未记录")
            reasons = ["命中兴趣词「" + "、".join(hits) + "」"] if hits else []
            if hosted:
                reasons.append("期刊摘要节选" if paper.get("abstract") else "仅题名")
            elif classification.get("decision"):
                reasons.append(
                    f"本方向初筛：{zh_auto_decision(classification['decision'])}（{evidence}）"
                )
            else:
                reasons.append(
                    "尚无本方向初筛 · " + ("有原始摘要" if paper.get("abstract") else "仅有题名")
                )
            if section == "待读书架" or revisiting:
                reasons.insert(0, "我的决定：" + zh_human_decision(paper["human_decision"]))
            if paper.get("added"):
                reasons.insert(0, f"{paper['added']} 首次收录")
            st.caption(" · ".join(reasons))
    if size < total and st.button(f"继续浏览（还有 {total - size} 篇）"):
        st.session_state[size_key] = size + 5
        st.rerun()
