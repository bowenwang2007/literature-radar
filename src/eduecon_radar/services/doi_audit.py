from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from eduecon_radar.services.doi import canonicalize_doi


def audit_doi_normalization(database: Path) -> dict[str, Any]:
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT id,doi,title FROM papers ORDER BY id").fetchall()
    categories: dict[str, list[int]] = {
        "invalid_doi": [],
        "doi_url_stored_as_doi": [],
        "case_duplicates": [],
        "whitespace_duplicates": [],
        "missing_doi": [],
    }
    canonical_groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        raw = row["doi"]
        if raw is None or not str(raw).strip():
            categories["missing_doi"].append(int(row["id"]))
            continue
        canonical = canonicalize_doi(str(raw))
        if canonical is None:
            categories["invalid_doi"].append(int(row["id"]))
            continue
        canonical_groups[canonical].append(row)
        if str(raw).casefold().startswith(("http://", "https://", "doi:")):
            categories["doi_url_stored_as_doi"].append(int(row["id"]))
        if str(raw) != str(raw).strip():
            categories["whitespace_duplicates"].append(int(row["id"]))
    collisions = []
    for canonical, group in canonical_groups.items():
        if len(group) < 2:
            continue
        raw_values = {str(item["doi"]) for item in group}
        collisions.append(
            {
                "canonical_doi": canonical,
                "paper_ids": [int(item["id"]) for item in group],
                "stored_values": sorted(raw_values),
            }
        )
        if len({value.casefold() for value in raw_values}) == 1:
            categories["case_duplicates"].extend(int(item["id"]) for item in group)
    return {
        "database": str(database.resolve()),
        "total_papers": len(rows),
        "canonical_collision": collisions,
        **{name: sorted(set(ids)) for name, ids in categories.items()},
    }


def generate_doi_repair_plan(database: Path) -> dict[str, Any]:
    audit = audit_doi_normalization(database)
    return {
        "mode": "read_only_plan",
        "database": audit["database"],
        "automatic_execution_allowed": False,
        "collision_count": len(audit["canonical_collision"]),
        "requires_human_review": audit,
    }
