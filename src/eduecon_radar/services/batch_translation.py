"""Batch Chinese title translation with strict per-item isolation.

The overnight contract requires titles to be translated in batches of 5-10 with
strict JSON output, paper_id mapping, per-item failure isolation, and a bounded
fallback to smaller batches. This module implements that path while the legacy
per-paper TranslationService remains available for compatibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from eduecon_radar.db import Database, utcnow
from eduecon_radar.services.deepseek import DeepSeekClient, DeepSeekError, DeepSeekResponse

LOGGER = logging.getLogger(__name__)

BATCH_TITLE_PROMPT_VERSION = "2026-07-31-batch-title-v1"
DEFAULT_BATCH_SIZE = 10
MAX_SPLIT_DEPTH = 2


class BatchTitleItem(BaseModel):
    paper_id: int
    chinese_title: str = ""
    validation_status: str = "ok"
    error_category: str | None = None
    error_message: str | None = None


class BatchTitleTranslationOutput(BaseModel):
    translations: list[BatchTitleItem] = Field(default_factory=list)


@dataclass(frozen=True)
class BatchTitleResult:
    considered: int
    translated: int
    pending: int
    failed: int
    ai_calls: int
    input_tokens: int
    output_tokens: int
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)


def _validated_chinese_title(original: str, translated: str) -> str:
    value = (translated or "").strip()
    if not value:
        raise ValueError("empty_title_translation")
    if value.casefold() == original.strip().casefold():
        raise ValueError("title_translation_copied_original")
    if not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", value):
        raise ValueError("title_translation_not_chinese")
    if len(value) > 400:
        raise ValueError("title_translation_too_long")
    return value


class BatchTitleTranslationService:
    """Translate missing English titles in batches with strict JSON recovery."""

    def __init__(
        self,
        database: Database,
        prompts_dir: Path,
        client: DeepSeekClient | None,
        model: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_split_depth: int = MAX_SPLIT_DEPTH,
        execution_guard: Callable[[], None] | None = None,
    ) -> None:
        if not 1 <= batch_size <= 10:
            raise ValueError("Batch title batch size must be between 1 and 10")
        self.database = database
        self.prompts_dir = prompts_dir
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.max_split_depth = max_split_depth
        self.execution_guard = execution_guard

    def run(
        self,
        paper_ids: Sequence[int],
        *,
        task_id: int | None = None,
    ) -> BatchTitleResult:
        normalized = tuple(dict.fromkeys(int(item) for item in paper_ids))
        if not normalized:
            return BatchTitleResult(0, 0, 0, 0, 0, 0, 0)
        rows = self._load_candidates(normalized)
        candidates: list[dict[str, object]] = []
        translated = 0
        pending = 0
        failed = 0
        reasons: list[str] = []
        for row in rows:
            paper_id = int(row["id"])
            title = str(row["title"] or "")
            source = row["title_translation_source"]
            existing = row["chinese_title"]
            if existing and source == "human_edited":
                translated += 1
                continue
            if existing and self._translation_matches_title(paper_id, row):
                translated += 1
                continue
            if not title:
                failed += 1
                reasons.append(f"paper {paper_id}: missing English title")
                self._record_failure(paper_id, "missing_source_title")
                continue
            candidates.append({"id": paper_id, "title": title})
        if candidates and self.client is None:
            pending += len(candidates)
            for item in candidates:
                self._record_pending(int(item["id"]))
            return BatchTitleResult(
                considered=len(normalized),
                translated=translated,
                pending=pending,
                failed=failed,
                ai_calls=0,
                input_tokens=0,
                output_tokens=0,
                failure_reasons=tuple(reasons),
            )
        ai_calls = input_tokens = output_tokens = 0
        missing: list[dict[str, object]] = []
        translated_in_batches = 0
        failed_in_batches = 0
        for batch in self._chunks(candidates, self.batch_size):
            batch = list(batch)
            result = self._translate_batch_with_fallback(
                batch,
                task_id=task_id,
                depth=0,
                reasons=reasons,
            )
            ai_calls += result["ai_calls"]
            input_tokens += result["input_tokens"]
            output_tokens += result["output_tokens"]
            translated_in_batches += int(result["translated"])
            failed_in_batches += int(result["failed"])
            missing.extend(result["missing"])
        # Retry papers missing from otherwise-valid batch responses in smaller batches.
        if missing and self.client is not None:
            for batch in self._chunks(missing, max(1, self.batch_size // 2)):
                result = self._translate_batch_with_fallback(
                    list(batch),
                    task_id=task_id,
                    depth=1,
                    reasons=reasons,
                )
                ai_calls += result["ai_calls"]
                input_tokens += result["input_tokens"]
                output_tokens += result["output_tokens"]
                translated_in_batches += int(result["translated"])
                failed_in_batches += int(result["failed"])
                missing = [
                    item
                    for item in missing
                    if any(
                        int(candidate["id"]) == int(item["id"])
                        for candidate in result["missing"]
                    )
                ]
        for item in missing:
            paper_id = int(item["id"])
            failed += 1
            reasons.append(f"paper {paper_id}: batch response missing paper_id")
            self._record_failure(paper_id, "batch_response_missing_paper_id")
        return BatchTitleResult(
            considered=len(normalized),
            translated=translated + translated_in_batches,
            pending=pending,
            failed=failed + failed_in_batches,
            ai_calls=ai_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            failure_reasons=tuple(dict.fromkeys(reasons)),
        )

    def run_batch_titles(
        self,
        paper_ids: Sequence[int],
        *,
        task_id: int | None = None,
    ) -> BatchTitleResult:
        """Compatibility alias used by the personal-mode pipeline."""
        return self.run(paper_ids, task_id=task_id)

    def _translate_batch_with_fallback(
        self,
        batch: list[dict[str, object]],
        *,
        task_id: int | None,
        depth: int,
        reasons: list[str],
    ) -> dict[str, object]:
        if not batch:
            return {
                "ai_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "missing": [],
                "translated": 0,
                "failed": 0,
            }
        try:
            self._guard()
            response = self._request_batch(batch, task_id=task_id)
            self._guard()
        except Exception as exc:
            reasons.append(_safe_failure(exc))
            if depth < self.max_split_depth and len(batch) > 1:
                mid = len(batch) // 2
                left = self._translate_batch_with_fallback(
                    batch[:mid], task_id=task_id, depth=depth + 1, reasons=reasons
                )
                right = self._translate_batch_with_fallback(
                    batch[mid:], task_id=task_id, depth=depth + 1, reasons=reasons
                )
                return {
                    "ai_calls": int(left["ai_calls"]) + int(right["ai_calls"]),
                    "input_tokens": int(left["input_tokens"]) + int(right["input_tokens"]),
                    "output_tokens": int(left["output_tokens"]) + int(right["output_tokens"]),
                    "missing": [*left["missing"], *right["missing"]],
                    "translated": int(left["translated"]) + int(right["translated"]),
                    "failed": int(left["failed"]) + int(right["failed"]),
                }
            for item in batch:
                paper_id = int(item["id"])
                reasons.append(f"paper {paper_id}: {_safe_failure(exc)}")
                self._record_failure(paper_id, "batch_call_failed")
            return {
                "ai_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "missing": [],
                "translated": 0,
                "failed": len(batch),
            }
        output = response.data
        assert isinstance(output, BatchTitleTranslationOutput)
        missing: list[dict[str, object]] = []
        translated_count = 0
        failed_count = 0
        by_id = {int(item.paper_id): item for item in output.translations}
        for item in batch:
            paper_id = int(item["id"])
            if paper_id not in by_id:
                missing.append(item)
                continue
            entry = by_id[paper_id]
            original = str(item["title"])
            if (
                entry.validation_status not in {"ok", ""}
                or (not entry.chinese_title and entry.error_category)
            ):
                failed_count += 1
                category = entry.error_category or "translation_failed"
                message = entry.error_message or category
                reasons.append(f"paper {paper_id}: {message}")
                self._record_failure(
                    paper_id,
                    category,
                    error_category=category,
                    error_message=message,
                )
                continue
            try:
                validated = _validated_chinese_title(original, entry.chinese_title)
            except ValueError as exc:
                reasons.append(f"paper {paper_id}: {exc}")
                failed_count += 1
                self._record_failure(
                    paper_id,
                    "validation_error",
                    error_category=str(exc),
                    error_message=str(exc),
                )
                continue
            translated_count += 1
            self._save_success(paper_id, original, validated, response)
        return {
            "ai_calls": 1,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "missing": missing,
            "translated": translated_count,
            "failed": failed_count,
        }

    def _request_batch(
        self, batch: list[dict[str, object]], *, task_id: int | None
    ) -> DeepSeekResponse:
        if self.client is None:
            raise DeepSeekError("AI translation client is not configured")
        system = (self.prompts_dir / "batch_title_system.txt").read_text(encoding="utf-8")
        user_template = (self.prompts_dir / "batch_title_user.txt").read_text(
            encoding="utf-8"
        )
        payload = [
            {"paper_id": int(item["id"]), "title": item["title"]} for item in batch
        ]
        user = user_template.format(
            items=json.dumps(payload, ensure_ascii=False),
            task_id=task_id or "",
        )
        return self.client.request_json(
            model=self.model,
            system_prompt=system,
            user_prompt=user,
            schema=BatchTitleTranslationOutput,
        )

    def _load_candidates(self, paper_ids: Sequence[int]) -> list[dict[str, object]]:
        placeholders = ",".join("?" for _ in paper_ids)
        sql = f"""SELECT p.id,p.title,pt.chinese_title,pt.title_translation_source,
        pt.title_translation_status,pt.source_title_hash
        FROM papers p LEFT JOIN paper_translations pt ON pt.paper_id=p.id
        WHERE p.id IN ({placeholders}) ORDER BY p.id"""
        with self.database.connect() as connection:
            rows = [dict(row) for row in connection.execute(sql, list(paper_ids)).fetchall()]
        return rows

    def _translation_matches_title(self, paper_id: int, row: dict[str, object]) -> bool:
        stored_hash = row.get("source_title_hash")
        if stored_hash:
            return str(stored_hash) == hashlib.sha256(
                str(row["title"] or "").encode("utf-8")
            ).hexdigest()
        # Legacy rows have no stored hash; keep the existing translation instead
        # of repeating an AI call when the paper title has not changed.
        with self.database.connect() as connection:
            metadata = connection.execute(
                "SELECT source_metadata_json FROM papers WHERE id=?", (paper_id,)
            ).fetchone()
        provenance = {}
        if metadata:
            provenance = json.loads(metadata[0] or "{}").get("title_translation_provenance")
        if isinstance(provenance, dict) and provenance.get("source_title_sha256"):
            return str(provenance["source_title_sha256"]) == hashlib.sha256(
                str(row["title"] or "").encode("utf-8")
            ).hexdigest()
        return True

    def _save_success(
        self, paper_id: int, original: str, chinese_title: str, response: DeepSeekResponse
    ) -> None:
        now = utcnow()
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO paper_translations
                (paper_id,chinese_title,title_translation_status,
                 abstract_translation_status,title_translation_source,
                 model_name,prompt_version,translated_at,updated_at,
                 source_title_hash,validation_status,error_category,error_message)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(paper_id) DO UPDATE SET
                  chinese_title=excluded.chinese_title,
                  title_translation_status=excluded.title_translation_status,
                  title_translation_source=excluded.title_translation_source,
                  model_name=excluded.model_name,prompt_version=excluded.prompt_version,
                  translated_at=excluded.translated_at,updated_at=excluded.updated_at,
                  source_title_hash=excluded.source_title_hash,
                  validation_status=excluded.validation_status,
                  error_category=excluded.error_category,
                  error_message=excluded.error_message""",
                (
                    paper_id,
                    chinese_title,
                    "ai_generated",
                    "unavailable",
                    "ai_generated",
                    self.model,
                    BATCH_TITLE_PROMPT_VERSION,
                    now,
                    now,
                    digest,
                    "ok",
                    None,
                    None,
                ),
            )
            row = connection.execute(
                "SELECT source_metadata_json FROM papers WHERE id=?", (paper_id,)
            ).fetchone()
            metadata = json.loads(row[0] or "{}") if row else {}
            metadata["title_translation_provenance"] = {
                "status": "current",
                "source_title_sha256": digest,
                "model_name": self.model,
                "prompt_version": BATCH_TITLE_PROMPT_VERSION,
                "translated_at": now,
                "source": "ai_generated",
            }
            connection.execute(
                "UPDATE papers SET source_metadata_json=? WHERE id=?",
                (json.dumps(metadata, ensure_ascii=False), paper_id),
            )

    def _record_failure(
        self,
        paper_id: int,
        category: str,
        *,
        error_category: str | None = None,
        error_message: str | None = None,
    ) -> None:
        now = utcnow()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO paper_translations
                (paper_id,chinese_title,title_translation_status,
                 abstract_translation_status,title_translation_source,
                 model_name,prompt_version,translated_at,updated_at,
                 source_title_hash,validation_status,error_category,error_message)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(paper_id) DO UPDATE SET
                  title_translation_status=excluded.title_translation_status,
                  model_name=excluded.model_name,prompt_version=excluded.prompt_version,
                  updated_at=excluded.updated_at,
                  validation_status=excluded.validation_status,
                  error_category=excluded.error_category,
                  error_message=excluded.error_message""",
                (
                    paper_id,
                    None,
                    "failed",
                    "unavailable",
                    None,
                    self.model,
                    BATCH_TITLE_PROMPT_VERSION,
                    now,
                    now,
                    None,
                    "error",
                    error_category or category,
                    error_message or category,
                ),
            )

    def _record_pending(self, paper_id: int) -> None:
        now = utcnow()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO paper_translations
                (paper_id,chinese_title,title_translation_status,
                 abstract_translation_status,title_translation_source,
                 model_name,prompt_version,translated_at,updated_at,
                 source_title_hash,validation_status,error_category,error_message)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(paper_id) DO UPDATE SET
                  title_translation_status=excluded.title_translation_status,
                  updated_at=excluded.updated_at,
                  validation_status=excluded.validation_status,
                  error_category=excluded.error_category,
                  error_message=excluded.error_message""",
                (
                    paper_id,
                    None,
                    "pending",
                    "unavailable",
                    None,
                    self.model,
                    BATCH_TITLE_PROMPT_VERSION,
                    now,
                    now,
                    None,
                    "pending",
                    "deepseek_api_key_missing",
                    "AI translation client is not configured",
                ),
            )

    @staticmethod
    def _chunks(values: Sequence[object], size: int) -> list[list[object]]:
        return [values[index : index + size] for index in range(0, len(values), size)]

    def _guard(self) -> None:
        if self.execution_guard is not None:
            self.execution_guard()


def _safe_failure(exc: Exception) -> str:
    return str(exc).replace("Authorization", "[redacted]")[:500]
