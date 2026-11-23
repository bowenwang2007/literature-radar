from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.archive import sha256_file

LABELS = ("include", "borderline", "exclude", "uncertain")


@dataclass(frozen=True)
class AnnotationProgress:
    total: int
    completed: int
    remaining: int
    label_counts: dict[str, int]
    marked_for_later: int


@dataclass(frozen=True)
class AgreementReport:
    paired_items: int
    raw_agreement: float | None
    cohen_kappa: float | None
    per_label_agreement: dict[str, float | None]
    include_exclude_direct_conflicts: int
    uncertain_proportion: float | None
    adjudication_rate: float | None
    subgroup_agreement: dict[str, dict[str, int | float | None]]
    ready_for_calibration: bool
    warnings: tuple[str, ...]


def init_gold_set_database(
    database_path: Path,
    migration_path: Path,
    *,
    production_database: Path,
    confirm_non_production_db: bool,
) -> Database:
    database_path = database_path.resolve()
    if database_path == production_database.resolve():
        raise ValueError("Gold Set schema cannot be initialized in the production database")
    if not confirm_non_production_db:
        raise ValueError("Use --confirm-non-production-db for Gold Set initialization")
    if database_path.exists():
        raise FileExistsError(f"Gold Set database already exists: {database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(migration_path.read_text(encoding="utf-8"))
        connection.commit()
    except Exception:
        connection.rollback()
        connection.close()
        database_path.unlink(missing_ok=True)
        raise
    connection.close()
    return Database(database_path)


def apply_gold_set_migration(
    database: Database,
    migration_path: Path,
    *,
    production_database: Path,
    confirm_non_production_db: bool,
) -> None:
    if database.path.resolve() == production_database.resolve():
        raise ValueError("Gold Set migration is forbidden on the production database")
    if not confirm_non_production_db:
        raise ValueError("Use --confirm-non-production-db for Gold Set migration")
    if not database.path.is_file():
        raise FileNotFoundError(f"Gold Set database does not exist: {database.path}")
    with database.connect() as connection:
        connection.executescript(migration_path.read_text(encoding="utf-8"))


def gold_set_schema_ready(database: Database) -> bool:
    if not database.path.is_file():
        return False
    uri = database.path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('gold_sets','gold_set_items','gold_set_annotations',"
                "'gold_set_adjudications','gold_set_deferrals','gold_set_snapshots',"
                "'calibration_runs',"
                "'model_bundle_candidates')"
            )
        }
        item_columns = {row[1] for row in connection.execute("PRAGMA table_info(gold_set_items)")}
    return len(names) == 8 and "formal_chinese_title" in item_columns


class GoldSetRepository:
    def __init__(self, database: Database) -> None:
        if not gold_set_schema_ready(database):
            raise RuntimeError("Gold Set schema is not installed")
        self.database = database

    def import_sample(
        self,
        *,
        items_csv: Path,
        sampling_manifest: Path,
        name: str,
        annotation_version: str,
        review_mode: str,
        minimum_round_interval_days: int | None = 7,
    ) -> int:
        if review_mode not in {"dual_reviewer", "single_reviewer_two_rounds"}:
            raise ValueError("Unsupported Gold Set review mode")
        manifest = json.loads(sampling_manifest.read_text(encoding="utf-8"))
        with items_csv.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        if not rows:
            raise ValueError("Gold Set sample is empty")
        if any((row.get("gold_label") or "").strip() for row in rows):
            raise ValueError("Imported Gold Set sample must have blank gold_label values")
        with self.database.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO gold_sets
                (gold_set_uid,name,profile_id,source_db_path,source_db_sha256,
                 sampling_manifest_json,annotation_version,review_mode,
                 minimum_round_interval_days,status,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,'annotating_round_1',?)""",
                (
                    manifest["gold_set_id"],
                    name.strip(),
                    "education_economics",
                    manifest["source_db_path"],
                    manifest["source_db_sha256"],
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                    annotation_version,
                    review_mode,
                    minimum_round_interval_days,
                    utcnow(),
                ),
            )
            gold_set_id = int(cursor.lastrowid)
            for row in rows:
                connection.execute(
                    """INSERT INTO gold_set_items
                    (gold_set_id,source_paper_id,doi,original_title,formal_chinese_title,
                     original_abstract,
                     authors_json,journal,journal_group,publication_date,keywords_json,
                     subject_codes_json,source_metadata_json,stratum,difficulty_tags_json,
                     display_order_seed,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        gold_set_id,
                        int(row["source_paper_id"]),
                        row.get("doi") or None,
                        row["original_title"],
                        row.get("formal_chinese_title") or None,
                        row.get("original_abstract") or None,
                        row.get("authors_json") or "[]",
                        row["journal"],
                        row["journal_group"],
                        row.get("publication_date") or None,
                        row.get("keywords_json") or "[]",
                        row.get("subject_codes_json") or "[]",
                        row.get("source_metadata_json") or "{}",
                        row["stratum"],
                        row.get("difficulty_tags_json") or "[]",
                        int(row["display_order_seed"]),
                        utcnow(),
                    ),
                )
        return gold_set_id

    def list_gold_sets(self) -> list[dict]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM gold_sets ORDER BY created_at DESC,id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def difficulty_tags(self, gold_set_id: int) -> tuple[str, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT difficulty_tags_json FROM gold_set_items WHERE gold_set_id=?",
                (gold_set_id,),
            ).fetchall()
        return tuple(
            sorted(
                {
                    str(tag)
                    for row in rows
                    for tag in json.loads(row["difficulty_tags_json"] or "[]")
                }
            )
        )

    def overview(self, gold_set_id: int) -> dict:
        with self.database.connect() as connection:
            gold_set = connection.execute(
                "SELECT * FROM gold_sets WHERE id=?", (gold_set_id,)
            ).fetchone()
            if not gold_set:
                raise ValueError("Gold Set not found")
            total = connection.execute(
                "SELECT COUNT(*) FROM gold_set_items WHERE gold_set_id=?", (gold_set_id,)
            ).fetchone()[0]
            annotations = connection.execute(
                """SELECT review_round,COUNT(*) count FROM gold_set_annotations a
                JOIN gold_set_items i ON i.id=a.gold_set_item_id
                WHERE i.gold_set_id=? GROUP BY review_round""",
                (gold_set_id,),
            ).fetchall()
            adjudicated = connection.execute(
                """SELECT COUNT(*) FROM gold_set_adjudications a JOIN gold_set_items i
                ON i.id=a.gold_set_item_id WHERE i.gold_set_id=?""",
                (gold_set_id,),
            ).fetchone()[0]
            snapshot = connection.execute(
                "SELECT * FROM gold_set_snapshots WHERE gold_set_id=? ORDER BY id DESC LIMIT 1",
                (gold_set_id,),
            ).fetchone()
        result = dict(gold_set)
        result.update(
            {
                "total_items": int(total),
                "round_1_completed": next(
                    (int(row["count"]) for row in annotations if row["review_round"] == 1), 0
                ),
                "round_2_completed": next(
                    (int(row["count"]) for row in annotations if row["review_round"] == 2), 0
                ),
                "adjudicated": int(adjudicated),
                "snapshot": dict(snapshot) if snapshot else None,
            }
        )
        return result

    def next_blind_item(
        self,
        *,
        gold_set_id: int,
        reviewer_id: str,
        review_round: int,
        difficulty_tag: str | None = None,
        include_deferred: bool = False,
        show_chinese_title: bool = False,
    ) -> dict | None:
        reviewer = reviewer_id.strip()
        if not reviewer:
            raise ValueError("reviewer_id is required")
        if review_round not in {1, 2}:
            raise ValueError("review_round must be 1 or 2")
        conditions = [
            "i.gold_set_id=?",
            "NOT EXISTS(SELECT 1 FROM gold_set_annotations a "
            "WHERE a.gold_set_item_id=i.id AND a.reviewer_id=? AND a.review_round=?)",
        ]
        values: list[object] = [gold_set_id, reviewer, review_round]
        if not include_deferred:
            conditions.append(
                "NOT EXISTS(SELECT 1 FROM gold_set_deferrals d "
                "WHERE d.gold_set_item_id=i.id AND d.reviewer_id=? AND d.review_round=?)"
            )
            values.extend((reviewer, review_round))
        if review_round == 2:
            conditions.append(
                "(SELECT g.review_mode FROM gold_sets g WHERE g.id=i.gold_set_id) "
                "='single_reviewer_two_rounds' AND EXISTS(SELECT 1 FROM "
                "gold_set_annotations first_round JOIN gold_sets g ON g.id=i.gold_set_id "
                "WHERE first_round.gold_set_item_id=i.id "
                "AND first_round.reviewer_id=? AND first_round.review_round=1 "
                "AND datetime(first_round.completed_at) <= datetime('now', "
                "printf('-%d days',COALESCE(g.minimum_round_interval_days,0))))"
            )
            values.append(reviewer)
        if difficulty_tag:
            conditions.append("i.difficulty_tags_json LIKE ?")
            values.append(f'%"{difficulty_tag}"%')
        with self.database.connect() as connection:
            row = connection.execute(
                f"""SELECT i.id,i.source_paper_id,i.doi,i.original_title,
                i.formal_chinese_title,
                i.original_abstract,i.authors_json,i.journal,i.journal_group,
                i.publication_date,i.keywords_json,i.subject_codes_json
                FROM gold_set_items i WHERE {" AND ".join(conditions)}
                ORDER BY ((i.display_order_seed + ? * 104729) % 2147483647),i.id LIMIT 1""",
                (*values, review_round),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        if not show_chinese_title:
            result.pop("formal_chinese_title", None)
        for name in ("authors_json", "keywords_json", "subject_codes_json"):
            result[name.removesuffix("_json")] = json.loads(result.pop(name) or "[]")
        result["blind_context"] = {
            "old_ai_visible": False,
            "rule_evidence_visible": False,
            "other_human_label_visible": False,
            "first_round_label_visible": False,
            "chinese_title_visible_by_default": False,
            "chinese_title_shown": show_chinese_title,
        }
        return result

    def defer_item(self, *, item_id: int, reviewer_id: str, review_round: int) -> int:
        reviewer = reviewer_id.strip()
        if not reviewer:
            raise ValueError("reviewer_id is required")
        if review_round not in {1, 2}:
            raise ValueError("review_round must be 1 or 2")
        with self.database.connect() as connection:
            mode = connection.execute(
                """SELECT g.review_mode,g.minimum_round_interval_days
                FROM gold_set_items i JOIN gold_sets g ON g.id=i.gold_set_id
                WHERE i.id=?""",
                (item_id,),
            ).fetchone()
            if mode is None:
                raise ValueError("Gold Set item not found")
            if review_round == 2:
                if mode["review_mode"] != "single_reviewer_two_rounds":
                    raise ValueError("Round 2 is only valid in single-reviewer two-round mode")
                eligible = connection.execute(
                    """SELECT 1 FROM gold_set_annotations WHERE gold_set_item_id=?
                    AND reviewer_id=? AND review_round=1
                    AND datetime(completed_at) <= datetime('now',printf('-%d days',?))""",
                    (item_id, reviewer, int(mode["minimum_round_interval_days"] or 0)),
                ).fetchone()
                if eligible is None:
                    raise ValueError("Round 2 interval or matching Round 1 requirement not met")
            cursor = connection.execute(
                """INSERT INTO gold_set_deferrals
                (gold_set_item_id,reviewer_id,review_round,deferred_at)
                VALUES (?,?,?,?) ON CONFLICT(gold_set_item_id,reviewer_id,review_round)
                DO UPDATE SET deferred_at=excluded.deferred_at""",
                (item_id, reviewer, review_round, utcnow()),
            )
        return int(cursor.lastrowid or 0)

    def submit_annotation(
        self,
        *,
        item_id: int,
        reviewer_id: str,
        review_round: int,
        label: str,
        reviewer_note: str | None = None,
        chinese_title_shown: bool = False,
        marked_for_later: bool = False,
        started_at: str | None = None,
    ) -> int:
        reviewer = reviewer_id.strip()
        if not reviewer:
            raise ValueError("reviewer_id is required")
        if label not in LABELS:
            raise ValueError(f"Unsupported Gold label: {label}")
        if review_round not in {1, 2}:
            raise ValueError("review_round must be 1 or 2")
        with self.database.connect() as connection:
            mode = connection.execute(
                """SELECT g.review_mode,g.minimum_round_interval_days
                FROM gold_set_items i JOIN gold_sets g ON g.id=i.gold_set_id
                WHERE i.id=?""",
                (item_id,),
            ).fetchone()
            if mode is None:
                raise ValueError("Gold Set item not found")
            if review_round == 2:
                if mode["review_mode"] != "single_reviewer_two_rounds":
                    raise ValueError("Round 2 is only valid in single-reviewer two-round mode")
                eligible = connection.execute(
                    """SELECT 1 FROM gold_set_annotations WHERE gold_set_item_id=?
                    AND reviewer_id=? AND review_round=1
                    AND datetime(completed_at) <= datetime('now',printf('-%d days',?))""",
                    (item_id, reviewer, int(mode["minimum_round_interval_days"] or 0)),
                ).fetchone()
                if eligible is None:
                    raise ValueError("Round 2 interval or matching Round 1 requirement not met")
            cursor = connection.execute(
                """INSERT INTO gold_set_annotations
                (gold_set_item_id,reviewer_id,review_round,label,reviewer_note,
                 chinese_title_shown,ai_info_shown,marked_for_later,started_at,completed_at)
                VALUES (?,?,?,?,?,?,0,?,?,?)""",
                (
                    item_id,
                    reviewer,
                    review_round,
                    label,
                    (reviewer_note or "").strip() or None,
                    int(chinese_title_shown),
                    int(marked_for_later),
                    started_at,
                    utcnow(),
                ),
            )
            connection.execute(
                """DELETE FROM gold_set_deferrals
                WHERE gold_set_item_id=? AND reviewer_id=? AND review_round=?""",
                (item_id, reviewer, review_round),
            )
        return int(cursor.lastrowid)

    def progress(
        self, *, gold_set_id: int, reviewer_id: str, review_round: int
    ) -> AnnotationProgress:
        with self.database.connect() as connection:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM gold_set_items WHERE gold_set_id=?", (gold_set_id,)
                ).fetchone()[0]
            )
            rows = connection.execute(
                """SELECT a.label,a.marked_for_later FROM gold_set_annotations a
                JOIN gold_set_items i ON i.id=a.gold_set_item_id
                WHERE i.gold_set_id=? AND a.reviewer_id=? AND a.review_round=?""",
                (gold_set_id, reviewer_id.strip(), review_round),
            ).fetchall()
            deferred = int(
                connection.execute(
                    """SELECT COUNT(*) FROM gold_set_deferrals d
                    JOIN gold_set_items i ON i.id=d.gold_set_item_id
                    WHERE i.gold_set_id=? AND d.reviewer_id=? AND d.review_round=?""",
                    (gold_set_id, reviewer_id.strip(), review_round),
                ).fetchone()[0]
            )
        counts = Counter(row["label"] for row in rows)
        return AnnotationProgress(
            total=total,
            completed=len(rows),
            remaining=max(0, total - len(rows)),
            label_counts={label: int(counts.get(label, 0)) for label in LABELS},
            marked_for_later=deferred + sum(int(row["marked_for_later"]) for row in rows),
        )

    def adjudication_queue(self, gold_set_id: int) -> list[dict]:
        pairs = self._annotation_pairs(gold_set_id)
        with self.database.connect() as connection:
            existing = {
                int(row["gold_set_item_id"])
                for row in connection.execute(
                    """SELECT a.gold_set_item_id FROM gold_set_adjudications a
                    JOIN gold_set_items i ON i.id=a.gold_set_item_id
                    WHERE i.gold_set_id=?""",
                    (gold_set_id,),
                )
            }
        queue = []
        for item_id, data in pairs.items():
            labels = [annotation["label"] for annotation in data["annotations"]]
            needs = len(labels) >= 2 and (len(set(labels)) > 1 or "uncertain" in labels)
            if needs and item_id not in existing:
                queue.append(data)
        return queue

    def adjudicate(
        self,
        *,
        item_id: int,
        label: str,
        reason: str,
        adjudicator_id: str,
    ) -> int:
        if label not in LABELS:
            raise ValueError("Invalid adjudicated label")
        if not reason.strip():
            raise ValueError("adjudication_reason is required")
        if not adjudicator_id.strip():
            raise ValueError("adjudicator_id is required")
        with self.database.connect() as connection:
            annotations = connection.execute(
                "SELECT label FROM gold_set_annotations WHERE gold_set_item_id=?",
                (item_id,),
            ).fetchall()
            labels = [str(row["label"]) for row in annotations]
            if len(labels) < 2 or (len(set(labels)) == 1 and "uncertain" not in labels):
                raise ValueError("Adjudication requires a human conflict or uncertain label")
            cursor = connection.execute(
                """INSERT INTO gold_set_adjudications
                (gold_set_item_id,adjudicated_label,adjudication_reason,
                 adjudicator_id,adjudicated_at) VALUES (?,?,?,?,?)""",
                (item_id, label, reason.strip(), adjudicator_id.strip(), utcnow()),
            )
        return int(cursor.lastrowid)

    def agreement(self, gold_set_id: int, *, minimum_raw_agreement: float = 0.7) -> AgreementReport:
        pairs = self._annotation_pairs(gold_set_id)
        paired = [data for data in pairs.values() if len(data["annotations"]) >= 2]
        observed = [
            (data["annotations"][0]["label"], data["annotations"][1]["label"]) for data in paired
        ]
        raw = _ratio(sum(left == right for left, right in observed), len(observed))
        kappa = _cohen_kappa(observed)
        per_label = {
            label: _ratio(
                sum(left == right == label for left, right in observed),
                sum(left == label or right == label for left, right in observed),
            )
            for label in LABELS
        }
        direct = sum({left, right} == {"include", "exclude"} for left, right in observed)
        uncertain = _ratio(
            sum(left == "uncertain" or right == "uncertain" for left, right in observed),
            len(observed),
        )
        queue = self.adjudication_queue(gold_set_id)
        with self.database.connect() as connection:
            adjudications = int(
                connection.execute(
                    """SELECT COUNT(*) FROM gold_set_adjudications a
                    JOIN gold_set_items i ON i.id=a.gold_set_item_id
                    WHERE i.gold_set_id=?""",
                    (gold_set_id,),
                ).fetchone()[0]
            )
        subgroup = _subgroup_agreement(paired)
        warnings: list[str] = []
        if not paired:
            warnings.append("no_paired_annotations")
        if len(paired) != len(pairs):
            warnings.append("incomplete_double_annotation")
        if raw is not None and raw < minimum_raw_agreement:
            warnings.append("raw_agreement_below_gate")
        if queue:
            warnings.append("unresolved_adjudications")
        final_labels = [
            _final_label(data["annotations"], data.get("adjudication")) for data in pairs.values()
        ]
        if any(label is None for label in final_labels):
            warnings.append("unresolved_gold_labels")
        if "uncertain" in final_labels:
            warnings.append("uncertain_not_eligible_for_calibration")
        return AgreementReport(
            paired_items=len(paired),
            raw_agreement=raw,
            cohen_kappa=kappa,
            per_label_agreement=per_label,
            include_exclude_direct_conflicts=direct,
            uncertain_proportion=uncertain,
            adjudication_rate=_ratio(adjudications, len(paired)),
            subgroup_agreement=subgroup,
            ready_for_calibration=bool(paired) and not warnings,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def export_snapshot(
        self,
        *,
        gold_set_id: int,
        snapshot_version: str,
        output: Path,
    ) -> dict:
        output = output.resolve()
        if output.exists():
            raise FileExistsError(f"Gold snapshot already exists: {output}")
        overview = self.overview(gold_set_id)
        pairs = self._annotation_pairs(gold_set_id)
        rows: list[dict] = []
        for item_id, data in sorted(pairs.items()):
            final = _final_label(data["annotations"], data.get("adjudication"))
            if final is None:
                raise RuntimeError(f"Gold item {item_id} has no resolved final label")
            item = data["item"]
            rows.append(
                {
                    "item_id": f"gold-item-{item_id}",
                    "doi": item["doi"],
                    "original_title": item["original_title"],
                    "original_abstract": item["original_abstract"],
                    "authors": json.loads(item["authors_json"]),
                    "journal": item["journal"],
                    "journal_group": item["journal_group"],
                    "publication_date": item["publication_date"],
                    "keywords": json.loads(item["keywords_json"]),
                    "subject_codes": json.loads(item["subject_codes_json"]),
                    "final_label": final,
                    "difficulty_tags": json.loads(item["difficulty_tags_json"]),
                    "source_db_sha256": overview["source_db_sha256"],
                    "annotation_version": overview["annotation_version"],
                    "gold_set_version": snapshot_version,
                }
            )
        payload = {
            "snapshot_status": "IMMUTABLE HUMAN GOLD TRUTH",
            "gold_set_uid": overview["gold_set_uid"],
            "snapshot_version": snapshot_version,
            "annotation_version": overview["annotation_version"],
            "source_db_sha256": overview["source_db_sha256"],
            "items": rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        digest = sha256_file(output)
        if not digest:  # pragma: no cover - defensive impossible branch
            raise RuntimeError("Snapshot hash failed")
        counts = dict(Counter(item["final_label"] for item in rows))
        try:
            with self.database.connect() as connection:
                cursor = connection.execute(
                    """INSERT INTO gold_set_snapshots
                    (gold_set_id,snapshot_version,annotation_version,snapshot_path,
                     snapshot_sha256,item_count,label_counts_json,source_db_sha256,
                     immutable,created_at) VALUES (?,?,?,?,?,?,?,?,1,?)""",
                    (
                        gold_set_id,
                        snapshot_version,
                        overview["annotation_version"],
                        str(output),
                        digest,
                        len(rows),
                        json.dumps(counts, sort_keys=True),
                        overview["source_db_sha256"],
                        utcnow(),
                    ),
                )
                connection.execute(
                    """UPDATE gold_sets SET status='frozen',frozen_at=?,snapshot_sha256=?
                    WHERE id=?""",
                    (utcnow(), digest, gold_set_id),
                )
        except Exception:
            output.unlink(missing_ok=True)
            raise
        return {
            "snapshot_id": int(cursor.lastrowid),
            "snapshot_path": str(output),
            "snapshot_sha256": digest,
            "item_count": len(rows),
            "label_counts": counts,
        }

    def add_source_verification(
        self,
        *,
        profile_id: str,
        entity_type: str,
        entity_id: str,
        source_type: str,
        source_title: str,
        publisher_or_organization: str | None,
        source_url: str | None,
        accessed_at: str,
        verification_status: str,
        reviewer_id: str,
        reviewer_note: str | None = None,
    ) -> int:
        if not reviewer_id.strip() or not source_title.strip() or not entity_id.strip():
            raise ValueError("Verification entity, source title and reviewer are required")
        with self.database.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO source_verifications
                (profile_id,entity_type,entity_id,source_type,source_title,
                 publisher_or_organization,source_url,accessed_at,verification_status,
                 reviewer_id,reviewer_note,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    profile_id,
                    entity_type,
                    entity_id,
                    source_type,
                    source_title,
                    publisher_or_organization,
                    source_url,
                    accessed_at,
                    verification_status,
                    reviewer_id.strip(),
                    reviewer_note,
                    utcnow(),
                ),
            )
        return int(cursor.lastrowid)

    def source_verification_summary(self, profile_id: str) -> dict:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT entity_type,verification_status,COUNT(*) count
                FROM source_verifications WHERE profile_id=?
                GROUP BY entity_type,verification_status""",
                (profile_id,),
            ).fetchall()
        return {
            f"{row['entity_type']}:{row['verification_status']}": int(row["count"]) for row in rows
        }

    def _annotation_pairs(self, gold_set_id: int) -> dict[int, dict]:
        with self.database.connect() as connection:
            items = connection.execute(
                "SELECT * FROM gold_set_items WHERE gold_set_id=? ORDER BY id", (gold_set_id,)
            ).fetchall()
            annotations = connection.execute(
                """SELECT a.* FROM gold_set_annotations a JOIN gold_set_items i
                ON i.id=a.gold_set_item_id WHERE i.gold_set_id=?
                ORDER BY a.gold_set_item_id,a.review_round,a.completed_at,a.id""",
                (gold_set_id,),
            ).fetchall()
            adjudications = connection.execute(
                """SELECT a.* FROM gold_set_adjudications a JOIN gold_set_items i
                ON i.id=a.gold_set_item_id WHERE i.gold_set_id=?""",
                (gold_set_id,),
            ).fetchall()
        result = {int(row["id"]): {"item": dict(row), "annotations": []} for row in items}
        for row in annotations:
            result[int(row["gold_set_item_id"])]["annotations"].append(dict(row))
        for row in adjudications:
            result[int(row["gold_set_item_id"])]["adjudication"] = dict(row)
        return result


def _final_label(annotations: list[dict], adjudication: dict | None) -> str | None:
    if adjudication:
        return str(adjudication["adjudicated_label"])
    if len(annotations) < 2:
        return None
    labels = [str(item["label"]) for item in annotations[:2]]
    return labels[0] if labels[0] == labels[1] and labels[0] != "uncertain" else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        left_counts[label] / len(pairs) * right_counts[label] / len(pairs) for label in LABELS
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def _subgroup_agreement(paired: list[dict]) -> dict[str, dict[str, int | float | None]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for data in paired:
        item = data["item"]
        tags = json.loads(item["difficulty_tags_json"] or "[]")
        for tag in tags:
            groups[f"difficulty:{tag}"].append(data)
        groups[f"journal_group:{item['journal_group']}"].append(data)
        abstract_group = "abstract:available" if item["original_abstract"] else "abstract:missing"
        groups[abstract_group].append(data)
    report: dict[str, dict[str, int | float | None]] = {}
    for name, members in sorted(groups.items()):
        agreements = sum(
            item["annotations"][0]["label"] == item["annotations"][1]["label"] for item in members
        )
        report[name] = {"count": len(members), "raw_agreement": _ratio(agreements, len(members))}
    return report
