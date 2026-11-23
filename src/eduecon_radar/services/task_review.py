from __future__ import annotations

import json
from dataclasses import dataclass

from eduecon_radar.db import Database, utcnow


@dataclass(frozen=True)
class TaskReviewInput:
    review_status: str
    final_decision: str | None = None
    reincluded: bool = False
    recovery_status: str = "not_applicable"
    reviewer: str | None = None
    reviewer_notes: str | None = None
    title_override: str | None = None
    abstract_override: str | None = None
    needs_edit: bool = False
    source: str = "v05_human"


def save_task_review(
    database: Database, task_id: int, paper_id: int, value: TaskReviewInput
) -> None:
    save_task_reviews(database, task_id, {paper_id: value})


def save_task_reviews(
    database: Database,
    task_id: int,
    values: dict[int, TaskReviewInput],
) -> None:
    if not values:
        return
    now = utcnow()
    paper_ids = tuple(dict.fromkeys(int(item) for item in values))
    placeholders = ",".join("?" for _ in paper_ids)
    with database.connect() as connection:
        rows = connection.execute(
            f"""SELECT wbi.paper_id,p.abstract
            FROM workflow_batch_items wbi JOIN papers p ON p.id=wbi.paper_id
            WHERE wbi.batch_id=? AND wbi.paper_id IN ({placeholders})""",  # noqa: S608
            (task_id, *paper_ids),
        ).fetchall()
        abstracts = {int(row["paper_id"]): row["abstract"] for row in rows}
        missing = set(paper_ids) - set(abstracts)
        if missing:
            if len(missing) == 1:
                paper_id = next(iter(missing))
                raise ValueError(
                    f"Paper {paper_id} does not belong to task {task_id}"
                )
            raise ValueError(
                f"Papers do not belong to task {task_id}: {sorted(missing)}"
            )
        encoded_rows: list[tuple[object, ...]] = []
        for paper_id, value in values.items():
            title_override = _override_text(value.title_override)
            abstract_override = _override_text(value.abstract_override)
            if abstract_override and not abstracts[paper_id]:
                raise ValueError("A task abstract override requires an original abstract")
            encoded_rows.append(
                (
                    task_id,
                    paper_id,
                    value.review_status,
                    int(value.reincluded),
                    value.recovery_status,
                    value.final_decision,
                    _text(value.reviewer),
                    _text(value.reviewer_notes),
                    title_override,
                    abstract_override,
                    int(value.needs_edit),
                    value.source,
                    json.dumps(
                        {"scope": "task", "task_id": task_id, "paper_id": paper_id},
                        sort_keys=True,
                    ),
                    now,
                    now,
                    now if value.review_status != "pending" else None,
                )
            )
        row_placeholders = "(" + ",".join("?" for _ in range(16)) + ")"
        values_sql = ",".join(row_placeholders for _ in encoded_rows)
        parameters = tuple(item for row in encoded_rows for item in row)
        connection.execute(
            f"""INSERT INTO workflow_task_reviews
            (task_id,paper_id,review_status,reincluded,recovery_status,final_decision,
             reviewer,reviewer_notes,title_override,abstract_override,needs_edit,source,
             provenance_json,created_at,updated_at,reviewed_at)
            VALUES {values_sql}
            ON CONFLICT(task_id,paper_id) DO UPDATE SET
              review_status=excluded.review_status,reincluded=excluded.reincluded,
              recovery_status=excluded.recovery_status,final_decision=excluded.final_decision,
              reviewer=excluded.reviewer,reviewer_notes=excluded.reviewer_notes,
              title_override=excluded.title_override,abstract_override=excluded.abstract_override,
              needs_edit=excluded.needs_edit,source=excluded.source,
              provenance_json=excluded.provenance_json,updated_at=excluded.updated_at,
              reviewed_at=excluded.reviewed_at""",  # noqa: S608
            parameters,
        )


def mark_task_recovery(
    database: Database, task_id: int, paper_id: int, *, recovered: bool
) -> None:
    existing = get_task_review(database, task_id, paper_id)
    review_status = (
        str(existing.get("review_status") or "pending") if existing else "pending"
    )
    save_task_review(
        database,
        task_id,
        paper_id,
        TaskReviewInput(
            review_status=review_status,
            final_decision=existing.get("final_decision") if existing else None,
            reincluded=recovered,
            recovery_status="recovered" if recovered else "not_recovered",
            reviewer=existing.get("reviewer") if existing else None,
            reviewer_notes=existing.get("reviewer_notes") if existing else None,
            title_override=existing.get("title_override") if existing else None,
            abstract_override=existing.get("abstract_override") if existing else None,
            needs_edit=bool(existing.get("needs_edit")) if existing else False,
            source="human_recovery" if recovered else "v05_human",
        ),
    )


def get_task_review(database: Database, task_id: int, paper_id: int) -> dict | None:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM workflow_task_reviews WHERE task_id=? AND paper_id=?",
            (task_id, paper_id),
        ).fetchone()
    return dict(row) if row else None


def task_review_counts(database: Database, task_id: int) -> dict[str, int]:
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT review_status,COUNT(*) count FROM workflow_task_reviews
            WHERE task_id=? GROUP BY review_status""",
            (task_id,),
        ).fetchall()
    return {str(row["review_status"]): int(row["count"]) for row in rows}


def _text(value: object) -> str | None:
    normalized = str(value).strip() if value is not None else ""
    return normalized or None


def _override_text(value: object) -> str | None:
    return str(value).strip() if value is not None else None
