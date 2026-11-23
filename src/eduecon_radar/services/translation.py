from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import AbstractTranslationResult, TitleTranslationResult
from eduecon_radar.services.deepseek import DeepSeekClient, DeepSeekError, DeepSeekResponse
from eduecon_radar.services.migrations import require_translation_schema

TRANSLATION_PROMPT_VERSION = "2026-07-19-translation-v1"


@dataclass(frozen=True)
class TranslationParameters:
    review_status: str | None = "approved"
    journal: str | None = None
    limit: int = 10
    missing_only: bool = True
    overwrite_human: bool = False
    paper_ids: tuple[int, ...] = ()
    translate_titles: bool = True
    translate_abstracts: bool = True
    task_id: int | None = None

    def __post_init__(self) -> None:
        if self.review_status not in {
            None,
            "pending",
            "approved",
            "rejected",
            "needs_revision",
        }:
            raise ValueError("Unsupported review status")
        if not 1 <= self.limit <= 100:
            raise ValueError("Translation limit must be between 1 and 100")
        normalized_ids = tuple(dict.fromkeys(int(item) for item in self.paper_ids))
        object.__setattr__(self, "paper_ids", normalized_ids)
        if not self.translate_titles and not self.translate_abstracts:
            raise ValueError("At least one translation field must be enabled")

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.review_status,
            self.journal or "",
            self.limit,
            self.missing_only,
            self.overwrite_human,
            self.paper_ids,
            self.translate_titles,
            self.translate_abstracts,
            self.task_id,
        )


@dataclass(frozen=True)
class TranslationPreview:
    candidates: tuple[dict, ...]
    matching_total: int
    title_pending: int
    abstract_pending: int
    abstract_unavailable: int


@dataclass
class TranslationStats:
    considered: int = 0
    titles_translated: int = 0
    abstracts_translated: int = 0
    unavailable: int = 0
    human_preserved: int = 0
    failed: int = 0
    ai_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float | None = None
    failure_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HumanTranslationSaveResult:
    title_changed: bool
    abstract_changed: bool

    @property
    def updated(self) -> bool:
        return self.title_changed or self.abstract_changed


def formal_translation_prefill(values: dict) -> tuple[str, str]:
    """Return only trusted formal translations; never use screening or legacy review fields."""

    def trusted(field: str, source_field: str, status_field: str) -> str:
        value = values.get(field)
        source = values.get(source_field)
        status = values.get(status_field)
        if value and (source == "human_edited" or status == "human_edited"):
            return str(value)
        if value and (source == "ai_generated" or status == "ai_generated"):
            return str(value)
        return ""

    return (
        trusted("formal_chinese_title", "title_translation_source", "title_translation_status"),
        trusted(
            "formal_chinese_abstract",
            "abstract_translation_source",
            "abstract_translation_status",
        ),
    )


def _candidate_conditions(parameters: TranslationParameters) -> tuple[list[str], list[object]]:
    conditions = ["1=1"]
    values: list[object] = []
    if parameters.review_status is not None:
        conditions.append("r.status=?")
        values.append(parameters.review_status)
    if parameters.journal:
        conditions.append("j.short_name=?")
        values.append(parameters.journal)
    if parameters.missing_only:
        missing = []
        if parameters.translate_titles:
            missing.append("pt.chinese_title IS NULL")
        if parameters.translate_abstracts:
            missing.append(
                "(p.abstract IS NOT NULL AND "
                "(pt.chinese_abstract IS NULL OR "
                "pt.abstract_translation_status IN ('pending','stale')))"
            )
        conditions.append(f"({' OR '.join(missing)})")
    if parameters.paper_ids:
        conditions.append(f"p.id IN ({','.join('?' for _ in parameters.paper_ids)})")
        values.extend(parameters.paper_ids)
    return conditions, values


def preview_translations(
    database: Database, parameters: TranslationParameters
) -> TranslationPreview:
    require_translation_schema(database)
    conditions, values = _candidate_conditions(parameters)
    sql = f"""SELECT p.id,p.title,p.abstract,j.name journal_name,j.short_name journal,
        pt.chinese_title,pt.chinese_abstract,pt.title_translation_source,
        pt.abstract_translation_source,pt.title_translation_status,
        pt.abstract_translation_status
        FROM papers p JOIN journals j ON j.id=p.journal_id
        JOIN reviews r ON r.paper_id=p.id
        LEFT JOIN paper_translations pt ON pt.paper_id=p.id
        WHERE {" AND ".join(conditions)} ORDER BY p.first_seen_at,p.id"""
    with database.connect() as connection:
        rows = [dict(row) for row in connection.execute(sql, values).fetchall()]
    selected = rows[: parameters.limit]
    return TranslationPreview(
        candidates=tuple(selected),
        matching_total=len(rows),
        title_pending=sum(not row["chinese_title"] for row in selected),
        abstract_pending=sum(
            bool(row["abstract"])
            and (
                not row["chinese_abstract"]
                or row["abstract_translation_status"] in {"pending", "stale"}
            )
            for row in selected
        ),
        abstract_unavailable=sum(not row["abstract"] for row in selected),
    )


def get_translation(database: Database, paper_id: int) -> dict | None:
    require_translation_schema(database)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM paper_translations WHERE paper_id=?", (paper_id,)
        ).fetchone()
    return dict(row) if row else None


def save_human_translation(
    database: Database,
    paper_id: int,
    *,
    chinese_title: str | None,
    chinese_abstract: str | None,
) -> HumanTranslationSaveResult:
    require_translation_schema(database)
    with database.connect() as connection:
        paper = connection.execute(
            """SELECT p.title,p.abstract,pt.id translation_id,pt.chinese_title,
            pt.chinese_abstract,pt.title_translation_status,pt.abstract_translation_status,
            pt.title_translation_source,pt.abstract_translation_source
            FROM papers p LEFT JOIN paper_translations pt ON pt.paper_id=p.id WHERE p.id=?""",
            (paper_id,),
        ).fetchone()
        if not paper:
            raise ValueError("Paper not found")
        title = (chinese_title or "").strip() or None
        abstract = (chinese_abstract or "").strip() or None
        if not paper["abstract"] and abstract:
            raise ValueError("原始摘要缺失，不能保存中文摘要翻译")
        title_changed = title != paper["chinese_title"]
        abstract_changed = abstract != paper["chinese_abstract"]
        result = HumanTranslationSaveResult(title_changed, abstract_changed)
        if not result.updated:
            return result
        now = utcnow()
        if paper["translation_id"] is not None:
            connection.execute(
                """UPDATE paper_translations SET chinese_title=?,chinese_abstract=?,
                title_translation_status=?,abstract_translation_status=?,
                title_translation_source=?,abstract_translation_source=?,
                translated_at=?,updated_at=? WHERE paper_id=?""",
                (
                    title,
                    abstract,
                    (
                        "human_edited"
                        if title_changed and title
                        else "pending"
                        if title_changed
                        else paper["title_translation_status"]
                    ),
                    (
                        "unavailable"
                        if not paper["abstract"]
                        else "human_edited"
                        if abstract_changed and abstract
                        else "pending"
                        if abstract_changed
                        else paper["abstract_translation_status"]
                    ),
                    (
                        "human_edited"
                        if title_changed and title
                        else None
                        if title_changed
                        else paper["title_translation_source"]
                    ),
                    (
                        "human_edited"
                        if abstract_changed and abstract
                        else None
                        if abstract_changed
                        else paper["abstract_translation_source"]
                    ),
                    now,
                    now,
                    paper_id,
                ),
            )
            return result
        connection.execute(
            """INSERT INTO paper_translations
            (paper_id,chinese_title,chinese_abstract,title_translation_status,
             abstract_translation_status,title_translation_source,abstract_translation_source,
             translated_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(paper_id) DO UPDATE SET
              chinese_title=excluded.chinese_title,
              chinese_abstract=excluded.chinese_abstract,
              title_translation_status=excluded.title_translation_status,
              abstract_translation_status=excluded.abstract_translation_status,
              title_translation_source=excluded.title_translation_source,
              abstract_translation_source=excluded.abstract_translation_source,
              translated_at=excluded.translated_at,updated_at=excluded.updated_at""",
            (
                paper_id,
                title,
                abstract,
                "human_edited" if title else "pending",
                (
                    "unavailable"
                    if not paper["abstract"]
                    else "human_edited"
                    if abstract
                    else "pending"
                ),
                "human_edited" if title else None,
                "human_edited" if abstract else None,
                now,
                now,
            ),
        )
        return result


if TYPE_CHECKING:
    from eduecon_radar.services.ai_budget import AIBudgetManager


class TranslationService:
    def __init__(
        self,
        database: Database,
        prompts_dir: Path,
        client: DeepSeekClient | None,
        model: str,
        execution_guard: Callable[[], None] | None = None,
        budget_manager: AIBudgetManager | None = None,
    ) -> None:
        self.database = database
        self.prompts_dir = prompts_dir
        self.client = client
        self.model = model
        self.execution_guard = execution_guard
        self.budget_manager = budget_manager

    def run(self, parameters: TranslationParameters) -> TranslationStats:
        preview = preview_translations(self.database, parameters)
        stats = TranslationStats()
        for candidate in preview.candidates:
            self._guard()
            stats.considered += 1
            self._translate_candidate(candidate, parameters, stats)
        return stats

    def _translate_candidate(
        self, candidate: dict, parameters: TranslationParameters, stats: TranslationStats
    ) -> None:
        title = candidate["chinese_title"]
        abstract = candidate["chinese_abstract"]
        title_source = candidate["title_translation_source"]
        abstract_source = candidate["abstract_translation_source"]
        title_status = (
            candidate["title_translation_status"]
            or title_source
            or ("ai_generated" if title else "pending")
        )
        abstract_status = (
            candidate["abstract_translation_status"]
            or abstract_source
            or ("ai_generated" if abstract else "pending")
        )
        failed = False
        title_was_translated = False
        abstract_was_translated = False
        if not parameters.overwrite_human and (
            title_source == "human_edited" or abstract_source == "human_edited"
        ):
            stats.human_preserved += 1

        should_translate_title = parameters.translate_titles and not (
            parameters.missing_only and title
        )
        if should_translate_title and not (
            title_source == "human_edited" and not parameters.overwrite_human
        ):
            try:
                self._guard()
                response = self._translate_title(
                    candidate["title"], candidate["id"], parameters.task_id
                )
                self._guard()
                self._count(stats, response)
                result = response.data
                assert isinstance(result, TitleTranslationResult)
                title = _validated_chinese_title(
                    candidate["title"], result.chinese_title
                )
                title_source = "ai_generated"
                title_status = "ai_generated"
                title_was_translated = True
                stats.titles_translated += 1
            except DeepSeekError as exc:
                title_status = "failed"
                failed = True
                stats.failure_reasons.append(_safe_failure(exc))

        if not candidate["abstract"]:
            abstract = None
            abstract_source = None
            abstract_status = "unavailable"
            stats.unavailable += 1
        elif (
            parameters.translate_abstracts
            and not (
                parameters.missing_only
                and abstract
                and abstract_status not in {"pending", "stale"}
            )
            and not (abstract_source == "human_edited" and not parameters.overwrite_human)
        ):
            try:
                self._guard()
                response = self._translate_abstract(
                    candidate["abstract"], candidate["id"], parameters.task_id
                )
                self._guard()
                self._count(stats, response)
                result = response.data
                assert isinstance(result, AbstractTranslationResult)
                abstract = result.chinese_abstract
                abstract_source = "ai_generated"
                abstract_status = "ai_generated"
                abstract_was_translated = True
                stats.abstracts_translated += 1
            except DeepSeekError as exc:
                abstract_status = "failed"
                failed = True
                stats.failure_reasons.append(_safe_failure(exc))

        if failed:
            stats.failed += 1
        self._guard()
        self._save(
            candidate["id"],
            chinese_title=title,
            chinese_abstract=abstract,
            title_status=title_status,
            abstract_status=abstract_status,
            title_source=title_source,
            abstract_source=abstract_source,
            title_input=(candidate["title"] if title_was_translated else None),
            abstract_input=(
                candidate["abstract"] if abstract_was_translated else None
            ),
        )

    def _translate_title(
        self, original_title: str, paper_id: int, task_id: int | None
    ) -> DeepSeekResponse:
        if self.client is None:
            raise DeepSeekError("AI translation client is not configured")
        system = self._read_prompt("translation_title_system.txt")
        user = self._read_prompt("translation_title_user.txt").format(
            original_title=original_title
        )
        return self._budgeted_request(
            task_id=task_id,
            paper_id=paper_id,
            stage="title_translation",
            system_prompt=system,
            user_prompt=user,
            output_limit=512,
            model=self.model,
            schema=TitleTranslationResult,
        )

    def _translate_abstract(
        self, original_abstract: str, paper_id: int, task_id: int | None
    ) -> DeepSeekResponse:
        if self.client is None:
            raise DeepSeekError("AI translation client is not configured")
        system = self._read_prompt("translation_abstract_system.txt")
        user = self._read_prompt("translation_abstract_user.txt").format(
            original_abstract=original_abstract
        )
        return self._budgeted_request(
            task_id=task_id,
            paper_id=paper_id,
            stage="abstract_translation",
            system_prompt=system,
            user_prompt=user,
            output_limit=4096,
            model=self.model,
            schema=AbstractTranslationResult,
        )

    def _budgeted_request(
        self,
        *,
        task_id: int | None,
        paper_id: int,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        output_limit: int,
        model: str,
        schema,
    ) -> DeepSeekResponse:
        if self.client is None:
            raise DeepSeekError("AI translation client is not configured")
        reservation = None
        if self.budget_manager is not None:
            # The optional reservation guard belongs to the outer (task) layer.
            # Importing it only when one was supplied keeps the product
            # translation path free of task/lease/budget concepts.
            from eduecon_radar.services.ai_budget import (
                BudgetBlocked,
                estimate_tokens,
            )

            if task_id is None:
                raise BudgetBlocked("task_id_required")
            estimated_input, estimated_output = estimate_tokens(
                system_prompt + user_prompt, output_limit
            )
            identity = hashlib.sha256(
                f"{task_id}|{paper_id}|{stage}|{model}|{user_prompt}".encode()
            ).hexdigest()
            reservation = self.budget_manager.reserve(
                task_id=task_id,
                stage=stage,
                provider="deepseek-compatible",
                model_name=model,
                idempotency_key=identity,
                estimated_input_tokens=estimated_input,
                estimated_output_tokens=estimated_output,
            )
            if reservation.already_settled:
                raise BudgetBlocked("settled_call_output_missing")
        try:
            response = self.client.request_json(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
            )
        except Exception as exc:
            if reservation is not None:
                self.budget_manager.settle(
                    reservation,
                    input_tokens=None,
                    output_tokens=None,
                    failed=True,
                    error_type=type(exc).__name__,
                )
            raise
        if reservation is not None:
            self.budget_manager.settle(
                reservation,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        return response

    def _read_prompt(self, name: str) -> str:
        return (self.prompts_dir / name).read_text(encoding="utf-8")

    @staticmethod
    def _count(stats: TranslationStats, response: DeepSeekResponse) -> None:
        stats.ai_calls += 1
        stats.input_tokens += response.input_tokens
        stats.output_tokens += response.output_tokens

    def _save(
        self,
        paper_id: int,
        *,
        chinese_title: str | None,
        chinese_abstract: str | None,
        title_status: str,
        abstract_status: str,
        title_source: str | None,
        abstract_source: str | None,
        title_input: str | None,
        abstract_input: str | None,
    ) -> None:
        now = utcnow()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO paper_translations
                (paper_id,chinese_title,chinese_abstract,title_translation_status,
                 abstract_translation_status,title_translation_source,abstract_translation_source,
                 model_name,prompt_version,translated_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(paper_id) DO UPDATE SET
                  chinese_title=excluded.chinese_title,
                  chinese_abstract=excluded.chinese_abstract,
                  title_translation_status=excluded.title_translation_status,
                  abstract_translation_status=excluded.abstract_translation_status,
                  title_translation_source=excluded.title_translation_source,
                  abstract_translation_source=excluded.abstract_translation_source,
                  model_name=excluded.model_name,prompt_version=excluded.prompt_version,
                  translated_at=excluded.translated_at,updated_at=excluded.updated_at""",
                (
                    paper_id,
                    chinese_title,
                    chinese_abstract,
                    title_status,
                    abstract_status,
                    title_source,
                    abstract_source,
                    self.model,
                    TRANSLATION_PROMPT_VERSION,
                    now,
                    now,
                ),
            )
            if title_input is not None:
                row = connection.execute(
                    "SELECT source_metadata_json FROM papers WHERE id=?", (paper_id,)
                ).fetchone()
                metadata = json.loads(row[0] or "{}") if row else {}
                metadata["title_translation_provenance"] = {
                    "status": "current",
                    "source_title_sha256": hashlib.sha256(
                        title_input.encode("utf-8")
                    ).hexdigest(),
                    "model_name": self.model,
                    "prompt_version": TRANSLATION_PROMPT_VERSION,
                    "translated_at": now,
                    "source": "ai_generated",
                }
                connection.execute(
                    "UPDATE papers SET source_metadata_json=? WHERE id=?",
                    (json.dumps(metadata, ensure_ascii=False), paper_id),
                )
            if abstract_input is not None:
                row = connection.execute(
                    "SELECT source_metadata_json FROM papers WHERE id=?", (paper_id,)
                ).fetchone()
                metadata = json.loads(row[0] or "{}") if row else {}
                metadata["abstract_translation_provenance"] = {
                    "status": "current",
                    "source_abstract_sha256": hashlib.sha256(
                        abstract_input.encode("utf-8")
                    ).hexdigest(),
                    "model_name": self.model,
                    "prompt_version": TRANSLATION_PROMPT_VERSION,
                    "translated_at": now,
                }
                connection.execute(
                    "UPDATE papers SET source_metadata_json=? WHERE id=?",
                    (json.dumps(metadata, ensure_ascii=False), paper_id),
                )

    def _guard(self) -> None:
        if self.execution_guard is not None:
            self.execution_guard()


def _safe_failure(exc: Exception) -> str:
    return str(exc).replace("Authorization", "[redacted]")[:500]


def _validated_chinese_title(original: str, translated: str) -> str:
    value = translated.strip()
    if not value:
        raise DeepSeekError("Title translation returned an empty value")
    if value.casefold() == original.strip().casefold():
        raise DeepSeekError("Title translation copied the original title")
    if not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", value):
        raise DeepSeekError("Title translation did not contain Chinese text")
    return value
