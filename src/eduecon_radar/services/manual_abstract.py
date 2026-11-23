"""Human-supplied original abstracts for the product path.

An English abstract is a *global paper fact*: pasting it must invalidate the
machine translation and record provider-independent provenance, exactly once,
for every profile.  This lived in the retired prototype workflow module, which
forced the product page to import the whole historical task pipeline to save a
text box.
"""

from __future__ import annotations

import json

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.metadata_provenance import (
    abstract_quality_flags,
    load_source_priorities,
    provenance_schema_ready,
    record_provenance,
)


def save_manual_abstract(
    database: Database,
    paper_id: int,
    abstract: str,
    source_url: str | None = None,
) -> dict:
    abstract = abstract.strip()
    if not abstract:
        return {"error": "摘要内容不能为空"}
    chinese_count = sum(1 for c in abstract if "\u4e00" <= c <= "\u9fff")
    if chinese_count > 0:
        return {"error": "请粘贴英文原始摘要，不要粘贴中文翻译"}
    if abstract.startswith("http"):
        return {"error": "内容看起来是URL，不是摘要"}
    now = utcnow()
    with database.connect() as connection:
        row = connection.execute(
            "SELECT abstract,source_metadata_json FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
        if not row:
            return {"error": f"论文 {paper_id} 不存在"}
        existing = json.loads(row["source_metadata_json"] or "{}")
        abstract_changed = (row["abstract"] or "") != abstract
        requested_url = source_url or ""
        if (
            not abstract_changed
            and existing.get("abstract_source") == "manual"
            and str(existing.get("abstract_manual_url") or "") == requested_url
        ):
            return {
                "success": True,
                "paper_id": paper_id,
                "abstract_changed": False,
                "skipped": "英文摘要内容和来源未变化",
            }
        existing["abstract_source"] = "manual"
        existing["abstract_manual_url"] = requested_url
        if abstract_changed:
            translation_provenance = existing.get("abstract_translation_provenance")
            translation_provenance = (
                dict(translation_provenance)
                if isinstance(translation_provenance, dict)
                else {}
            )
            translation_provenance["status"] = "stale"
            translation_provenance["stale_at"] = now
            existing["abstract_translation_provenance"] = translation_provenance
        connection.execute(
            "UPDATE papers SET abstract=?, source_metadata_json=?, updated_at=? WHERE id=?",
            (abstract, json.dumps(existing, ensure_ascii=False), now, paper_id),
        )
        if abstract_changed:
            connection.execute(
                """UPDATE paper_translations
                SET abstract_translation_status='pending',
                    updated_at=?
                WHERE paper_id=?""",
                (now, paper_id),
            )
        if provenance_schema_ready(connection):
            _, priorities = load_source_priorities()
            record_provenance(
                connection,
                paper_id=paper_id,
                field_name="abstract",
                value=abstract,
                source_name="manual",
                source_record_id=source_url,
                source_priority=priorities["manual"],
                decision="accepted",
                reason="human_supplied_original_abstract",
                quality_flags=abstract_quality_flags(abstract),
                is_current=True,
                observed_at=now,
            )
    return {"success": True, "paper_id": paper_id, "abstract_changed": abstract_changed}
