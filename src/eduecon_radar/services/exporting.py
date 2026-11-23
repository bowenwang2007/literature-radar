from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from datetime import date

from eduecon_radar.db import Database
from eduecon_radar.services.migrations import require_p0_hardening_schema, translation_schema_ready
from eduecon_radar.services.review import LATEST_SCREENING_JOIN


@dataclass(frozen=True)
class MentorExportRow:
    id: int
    doi: str | None
    original_title: str
    chinese_title: str | None
    authors: tuple[str, ...]
    journal: str
    publication_date: str | None
    original_abstract: str | None
    chinese_abstract: str | None
    source_url: str | None
    open_access_url: str | None
    title_translation_source: str | None
    abstract_translation_source: str | None
    review_status: str
    human_note: str | None


@dataclass(frozen=True)
class PersonalResearchExportRow:
    id: int
    doi: str | None
    original_title: str
    authors: tuple[str, ...]
    journal: str
    publication_date: str | None
    source_url: str | None
    open_access_url: str | None
    ai_chinese_title: str | None
    ai_research_question: str | None
    ai_data_and_sample: str | None
    ai_identification_strategy: str | None
    ai_main_result: str | None
    ai_chinese_summary: str | None
    ai_topics: tuple[str, ...]
    ai_screening_reason: str | None
    human_edited_chinese_title: str | None
    human_edited_chinese_summary: str | None
    human_personal_note: str | None
    review_status: str


def _filters(
    *,
    review_status: str | tuple[str, ...] = "approved",
    journal: str | None = None,
    topic: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    has_abstract: bool | None = None,
    batch_id: int | None = None,
    review_alias: str = "r",
) -> tuple[list[str], list[object]]:
    statuses = (review_status,) if isinstance(review_status, str) else review_status
    if not statuses:
        raise ValueError("At least one review status is required")
    status_field = "review_status" if review_alias == "tr" else "status"
    conditions = [
        f"{review_alias}.{status_field} IN ({','.join('?' for _ in statuses)})"
    ]
    params: list[object] = list(statuses)
    if journal:
        conditions.append("j.short_name=?")
        params.append(journal)
    if topic:
        conditions.append(
            "EXISTS (SELECT 1 FROM screenings sf WHERE sf.paper_id=p.id AND sf.topics_json LIKE ?)"
        )
        params.append(f"%{topic}%")
    if date_from:
        conditions.append("COALESCE(p.issue_date,p.online_date)>=?")
        params.append(date_from.isoformat())
    if date_to:
        conditions.append("COALESCE(p.issue_date,p.online_date)<=?")
        params.append(date_to.isoformat())
    if has_abstract is not None:
        conditions.append("p.abstract IS NOT NULL" if has_abstract else "p.abstract IS NULL")
    if batch_id is not None and review_alias != "tr":
        conditions.append(
            "EXISTS (SELECT 1 FROM workflow_batch_items wbi "
            "WHERE wbi.paper_id=p.id AND wbi.batch_id=?)"
        )
        params.append(batch_id)
    return conditions, params


def mentor_records(
    database: Database,
    *,
    review_status: str | tuple[str, ...] = "approved",
    journal: str | None = None,
    topic: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    has_abstract: bool | None = None,
    batch_id: int | None = None,
    task_id: int | None = None,
    allow_pilot_task: bool = False,
) -> list[MentorExportRow]:
    if batch_id is not None and task_id is not None:
        raise ValueError("Use either legacy batch_id or v0.5 task_id, not both")
    task_scoped = task_id is not None
    if (
        task_scoped
        and _is_controlled_pilot_task(database, int(task_id))
        and not allow_pilot_task
    ):
        return []
    if task_scoped:
        require_p0_hardening_schema(database)
    conditions, params = _filters(
        review_status=review_status,
        journal=journal,
        topic=topic,
        date_from=date_from,
        date_to=date_to,
        has_abstract=has_abstract,
        batch_id=None if task_scoped else batch_id,
        review_alias="tr" if task_scoped else "r",
    )
    if translation_schema_ready(database):
        translation_join = "LEFT JOIN paper_translations pt ON pt.paper_id=p.id"
        translation_fields = """pt.chinese_title,pt.chinese_abstract,
            pt.title_translation_source,pt.abstract_translation_source"""
    else:
        translation_join = ""
        translation_fields = """NULL chinese_title,NULL chinese_abstract,
            NULL title_translation_source,NULL abstract_translation_source"""
    if task_scoped:
        assert task_id is not None
        review_join = """JOIN workflow_batch_items wbi ON wbi.paper_id=p.id
        JOIN workflow_task_reviews tr
          ON tr.task_id=wbi.batch_id AND tr.paper_id=wbi.paper_id AND tr.task_id=?"""
        params = [task_id, *params]
        review_fields = """tr.review_status review_status,
        tr.reviewer_notes human_note"""
        if translation_schema_ready(database):
            translation_fields = """COALESCE(tr.title_override,pt.chinese_title) chinese_title,
                COALESCE(tr.abstract_override,pt.chinese_abstract) chinese_abstract,
                CASE WHEN tr.title_override IS NOT NULL THEN 'task_human_override'
                     ELSE pt.title_translation_source END title_translation_source,
                CASE WHEN tr.abstract_override IS NOT NULL THEN 'task_human_override'
                     ELSE pt.abstract_translation_source END abstract_translation_source"""
    else:
        review_join = "JOIN reviews r ON r.paper_id=p.id"
        review_fields = "r.status review_status,r.reviewer_note human_note"
    sql = f"""SELECT p.id,p.doi,p.title original_title,p.authors_json,j.name journal,
        COALESCE(p.issue_date,p.online_date) publication_date,p.abstract original_abstract,
        p.source_url,p.open_access_url,{translation_fields},
        {review_fields}
        FROM papers p JOIN journals j ON j.id=p.journal_id
        {review_join} {translation_join}
        WHERE {" AND ".join(conditions)} ORDER BY publication_date DESC,p.id"""
    with database.connect() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [
        MentorExportRow(
            id=int(row["id"]),
            doi=row["doi"],
            original_title=row["original_title"],
            chinese_title=row["chinese_title"],
            authors=tuple(json.loads(row["authors_json"])),
            journal=row["journal"],
            publication_date=row["publication_date"],
            original_abstract=row["original_abstract"],
            chinese_abstract=row["chinese_abstract"],
            source_url=row["source_url"],
            open_access_url=row["open_access_url"],
            title_translation_source=row["title_translation_source"],
            abstract_translation_source=row["abstract_translation_source"],
            review_status=row["review_status"],
            human_note=row["human_note"],
        )
        for row in rows
    ]


def personal_research_records(
    database: Database,
    *,
    review_status: str | tuple[str, ...] = "approved",
    journal: str | None = None,
    topic: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    has_abstract: bool | None = None,
    batch_id: int | None = None,
    task_id: int | None = None,
    allow_pilot_task: bool = False,
) -> list[PersonalResearchExportRow]:
    if batch_id is not None and task_id is not None:
        raise ValueError("Use either legacy batch_id or v0.5 task_id, not both")
    task_scoped = task_id is not None
    if (
        task_scoped
        and _is_controlled_pilot_task(database, int(task_id))
        and not allow_pilot_task
    ):
        return []
    if task_scoped:
        require_p0_hardening_schema(database)
    conditions, params = _filters(
        review_status=review_status,
        journal=journal,
        topic=topic,
        date_from=date_from,
        date_to=date_to,
        has_abstract=has_abstract,
        batch_id=None if task_scoped else batch_id,
        review_alias="tr" if task_scoped else "r",
    )
    if task_scoped:
        assert task_id is not None
        review_join = """JOIN workflow_batch_items wbi ON wbi.paper_id=p.id
        JOIN workflow_task_reviews tr
          ON tr.task_id=wbi.batch_id AND tr.paper_id=wbi.paper_id AND tr.task_id=?"""
        params = [task_id, *params]
        human_fields = """tr.title_override human_edited_chinese_title,
        tr.abstract_override human_edited_chinese_summary,
        tr.reviewer_notes human_personal_note,tr.review_status review_status"""
    else:
        review_join = "JOIN reviews r ON r.paper_id=p.id"
        human_fields = """r.edited_chinese_title human_edited_chinese_title,
        r.edited_chinese_summary human_edited_chinese_summary,
        r.reviewer_note human_personal_note,r.status review_status"""
    sql = f"""SELECT p.id,p.doi,p.title original_title,p.authors_json,j.name journal,
        COALESCE(p.issue_date,p.online_date) publication_date,p.source_url,p.open_access_url,
        s.chinese_title ai_chinese_title,s.research_question ai_research_question,
        s.data_and_sample ai_data_and_sample,
        COALESCE(s.identification_strategy,s.method) ai_identification_strategy,
        s.main_result ai_main_result,s.chinese_summary ai_chinese_summary,
        s.topics_json,s.ai_reason ai_screening_reason,
        {human_fields}
        FROM papers p JOIN journals j ON j.id=p.journal_id
        {LATEST_SCREENING_JOIN} {review_join}
        WHERE {" AND ".join(conditions)} ORDER BY publication_date DESC,p.id"""
    with database.connect() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [
        PersonalResearchExportRow(
            id=int(row["id"]),
            doi=row["doi"],
            original_title=row["original_title"],
            authors=tuple(json.loads(row["authors_json"])),
            journal=row["journal"],
            publication_date=row["publication_date"],
            source_url=row["source_url"],
            open_access_url=row["open_access_url"],
            ai_chinese_title=row["ai_chinese_title"],
            ai_research_question=row["ai_research_question"],
            ai_data_and_sample=row["ai_data_and_sample"],
            ai_identification_strategy=row["ai_identification_strategy"],
            ai_main_result=row["ai_main_result"],
            ai_chinese_summary=row["ai_chinese_summary"],
            ai_topics=tuple(json.loads(row["topics_json"] or "[]")),
            ai_screening_reason=row["ai_screening_reason"],
            human_edited_chinese_title=row["human_edited_chinese_title"],
            human_edited_chinese_summary=row["human_edited_chinese_summary"],
            human_personal_note=row["human_personal_note"],
            review_status=row["review_status"],
        )
        for row in rows
    ]


def export_mentor_markdown(
    database: Database,
    *,
    include_human_note: bool = False,
    pilot_not_released: bool = False,
    **filters: object,
) -> str:
    records = mentor_records(database, **filters)
    notice = (
        "# CONTROLLED PILOT — NOT RELEASED — 不得作为正式筛选结果\n\n"
        if pilot_not_released
        else ""
    )
    if not records:
        return notice + "# EduEcon Radar 导师版\n\n当前筛选条件下没有可导出记录。\n"
    sections = [
        *(
            ["# CONTROLLED PILOT — NOT RELEASED — 不得作为正式筛选结果", ""]
            if pilot_not_released
            else []
        ),
        "# EduEcon Radar 导师版",
        "",
        "中文标题与中文摘要来自正式翻译层，可能经过人工校订；原版摘要来自Crossref元数据。",
        "",
    ]
    for item in records:
        sections.extend(
            [
                f"# {item.original_title}",
                "",
                "## 论文中文标题",
                "",
                item.chinese_title or "Chinese title translation unavailable.",
                "",
                f"**作者：** {', '.join(item.authors) or 'Unknown'}",
                f"**期刊：** {item.journal}",
                f"**发表日期：** {item.publication_date or 'Unavailable'}",
                f"**DOI：** {item.doi or 'Unavailable'}",
                f"**Source Link：** {item.source_url or 'Unavailable'}",
            ]
        )
        if item.open_access_url:
            sections.append(f"**Open Access Link：** {item.open_access_url}")
        sections.extend(
            [
                "",
                "## Original Abstract / 论文原版摘要",
                "",
                item.original_abstract or "Original abstract unavailable in current metadata.",
                "",
                "## Chinese Abstract Translation / 中文摘要翻译",
                "",
                item.chinese_abstract or "Chinese translation unavailable.",
                "",
            ]
        )
        if include_human_note:
            sections.extend(["## Human Note / 人工备注", "", item.human_note or "", ""])
    return "\n".join(sections)


def export_mentor_csv_text(
    database: Database,
    *,
    include_human_note: bool = False,
    pilot_not_released: bool = False,
    **filters: object,
) -> str:
    records = mentor_records(database, **filters)
    fields = [
        "id",
        "doi",
        "original_title",
        "chinese_title",
        "authors",
        "journal",
        "publication_date",
        "original_abstract",
        "chinese_abstract",
        "source_url",
        "open_access_url",
        "title_translation_source",
        "abstract_translation_source",
    ]
    if pilot_not_released:
        fields.insert(0, "pilot_notice")
    if include_human_note:
        fields.extend(["review_status", "human_note"])
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for item in records:
        row = asdict(item)
        row["authors"] = "; ".join(item.authors)
        if pilot_not_released:
            row["pilot_notice"] = "CONTROLLED PILOT — NOT RELEASED"
        writer.writerow({field: row[field] for field in fields})
    return "\ufeff" + output.getvalue()


def export_personal_markdown(
    database: Database, *, pilot_not_released: bool = False, **filters: object
) -> str:
    records = personal_research_records(database, **filters)
    notice = (
        "# CONTROLLED PILOT — NOT RELEASED — 不得作为正式筛选结果\n\n"
        if pilot_not_released
        else ""
    )
    if not records:
        return notice + "# Personal Research Export\n\n当前筛选条件下没有可导出记录。\n"
    sections = [
        *(
            ["# CONTROLLED PILOT — NOT RELEASED — 不得作为正式筛选结果", ""]
            if pilot_not_released
            else []
        ),
        "# Personal Research Export / 个人研究记录",
        "",
        "以下内容包含AI总结、分析和人工笔记，不是论文原始摘要或其正式翻译。",
        "",
    ]
    for item in records:
        sections.extend(
            [
                f"## {item.original_title}",
                "",
                f"- Authors: {', '.join(item.authors)}",
                f"- Journal: {item.journal}",
                f"- AI Chinese title: {item.ai_chinese_title or ''}",
                f"- AI research question: {item.ai_research_question or ''}",
                f"- AI data and sample: {item.ai_data_and_sample or ''}",
                f"- AI identification strategy: {item.ai_identification_strategy or ''}",
                f"- AI main result: {item.ai_main_result or ''}",
                f"- AI Chinese summary: {item.ai_chinese_summary or ''}",
                f"- AI topics: {'; '.join(item.ai_topics)}",
                f"- AI screening reason: {item.ai_screening_reason or ''}",
                f"- Human edited Chinese title: {item.human_edited_chinese_title or ''}",
                f"- Human edited Chinese summary: {item.human_edited_chinese_summary or ''}",
                f"- Human personal note: {item.human_personal_note or ''}",
                "",
            ]
        )
    return "\n".join(sections)


def export_personal_csv_text(
    database: Database, *, pilot_not_released: bool = False, **filters: object
) -> str:
    records = personal_research_records(database, **filters)
    fields = list(PersonalResearchExportRow.__dataclass_fields__)
    if pilot_not_released:
        fields.insert(0, "pilot_notice")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for item in records:
        row = asdict(item)
        row["authors"] = "; ".join(item.authors)
        row["ai_topics"] = "; ".join(item.ai_topics)
        if pilot_not_released:
            row["pilot_notice"] = "CONTROLLED PILOT — NOT RELEASED"
        writer.writerow(row)
    return "\ufeff" + output.getvalue()


# Backward-compatible function names now point to the safe mentor export.
approved_records = mentor_records
export_markdown = export_mentor_markdown
export_csv_text = export_mentor_csv_text


def _is_controlled_pilot_task(database: Database, task_id: int) -> bool:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT source_config_json FROM workflow_batches WHERE id=?", (task_id,)
        ).fetchone()
    if not row:
        return False
    try:
        config = json.loads(row[0] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return config.get("runtime_mode") == "controlled_pilot"
