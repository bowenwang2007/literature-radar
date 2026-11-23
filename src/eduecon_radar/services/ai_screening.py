# --- not part of the product core: see docs/PRODUCT_CORE.md ---
# LEGACY (CLI only). Writes the retired ``screenings`` table; the product
# screening path is ``services/deterministic_screening.py`` writing
# ``paper_screenings`` / the profile assessment tables.

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from eduecon_radar.db import Database, utcnow
from eduecon_radar.schemas import AIScreeningResult, PaperMetadata, RuleResult, SummaryResult
from eduecon_radar.services.deepseek import DeepSeekClient, DeepSeekError, DeepSeekResponse
from eduecon_radar.services.rule_screening import screen_by_rules

PROMPT_VERSION = "2026-07-18-v1"


@dataclass
class ScreeningStats:
    considered: int = 0
    screened: int = 0
    excluded_by_rules: int = 0
    unprocessed: int = 0
    ai_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def _paper_from_row(row: sqlite3.Row) -> PaperMetadata:
    return PaperMetadata(
        doi=row["doi"],
        title=row["title"],
        abstract=row["abstract"],
        authors=json.loads(row["authors_json"]),
        publisher=row["publisher"],
        article_type=row["article_type"],
        online_date=row["online_date"],
        issue_date=row["issue_date"],
        volume=row["volume"],
        issue=row["issue"],
        pages=row["pages"],
        source_url=row["source_url"],
        open_access_url=row["open_access_url"],
        keywords=json.loads(row["keywords_json"]),
        jel_codes=json.loads(row["jel_codes_json"]),
        source_metadata=json.loads(row["source_metadata_json"]),
    )


class ScreeningService:
    def __init__(
        self,
        database: Database,
        prompts_dir: Path,
        client: DeepSeekClient | None,
        fast_model: str,
        strong_model: str,
    ) -> None:
        self.database = database
        self.prompts_dir = prompts_dir
        self.client = client
        self.fast_model = fast_model
        self.strong_model = strong_model

    def run(
        self,
        *,
        limit: int | None = None,
        no_ai: bool = False,
        allow_strong: bool = True,
        paper_id: int | None = None,
        paper_ids: list[int] | tuple[int, ...] | None = None,
        force_strong: bool = False,
        force_rescreen: bool = False,
        source: str = "eligible",
    ) -> ScreeningStats:
        if source not in {"eligible", "awaiting_screening", "screening_failed"}:
            raise ValueError("Unsupported screening source")
        stats = ScreeningStats()
        params: list[object] = []
        sql = "SELECT p.*,j.name AS journal_name FROM papers p JOIN journals j ON j.id=p.journal_id"
        if paper_id is not None and paper_ids is not None:
            raise ValueError("paper_id and paper_ids cannot be used together")
        if paper_ids is not None:
            selected_ids = list(dict.fromkeys(int(item) for item in paper_ids))
            if not selected_ids:
                return stats
            placeholders = ",".join("?" for _ in selected_ids)
            sql += f" WHERE p.id IN ({placeholders})"
            params = selected_ids
            sql += " ORDER BY p.first_seen_at ASC"
        elif paper_id is None:
            if source == "awaiting_screening":
                sql += " WHERE NOT EXISTS (SELECT 1 FROM screenings s WHERE s.paper_id=p.id)"
            elif source == "screening_failed":
                sql += """ WHERE 'unprocessed'=(
                    SELECT s.ai_decision FROM screenings s WHERE s.paper_id=p.id
                    ORDER BY s.screened_at DESC,s.id DESC LIMIT 1
                )"""
            else:
                sql += """ WHERE NOT EXISTS (
                    SELECT 1 FROM screenings s
                    WHERE s.paper_id=p.id AND s.prompt_version=?
                      AND s.ai_decision != 'unprocessed'
                )"""
                params.append(PROMPT_VERSION)
            sql += " ORDER BY p.first_seen_at ASC"
        else:
            sql += " WHERE p.id=?"
            params = [paper_id]
        del force_rescreen  # eligibility is decided by the caller when exact IDs are supplied
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self.database.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        for row in rows:
            stats.considered += 1
            paper = _paper_from_row(row)
            rule = screen_by_rules(paper)
            if not rule.should_screen:
                self._save(
                    row["id"],
                    rule,
                    "rules-only",
                    "exclude",
                    1.0,
                    "No education-economics terms found by the broad rule screen",
                )
                stats.excluded_by_rules += 1
                continue
            if no_ai or self.client is None:
                self._save(
                    row["id"],
                    rule,
                    "rules-only",
                    "unprocessed",
                    None,
                    "AI disabled; metadata retained for later screening",
                )
                stats.unprocessed += 1
                continue
            try:
                selected_model = self.strong_model if force_strong else self.fast_model
                classification = self._classify(paper, row["journal_name"], selected_model)
                self._count(stats, classification)
                result = classification.data
                assert isinstance(result, AIScreeningResult)
                conflict = result.decision == "exclude" and rule.score >= 2.5
                if (
                    not force_strong
                    and allow_strong
                    and (0.40 <= result.confidence <= 0.75 or conflict)
                ):
                    classification = self._classify(paper, row["journal_name"], self.strong_model)
                    self._count(stats, classification)
                    result = classification.data
                    assert isinstance(result, AIScreeningResult)
                    selected_model = self.strong_model
                summary = None
                if result.decision in {"include", "borderline"}:
                    summary_response = self._summarize(paper, result, selected_model)
                    self._count(stats, summary_response)
                    summary = summary_response.data
                    assert isinstance(summary, SummaryResult)
                self._save_result(row["id"], rule, selected_model, result, summary)
                stats.screened += 1
            except DeepSeekError as exc:
                self._save(row["id"], rule, self.fast_model, "unprocessed", None, str(exc))
                stats.unprocessed += 1
        self._record_usage(stats)
        return stats

    def _read_prompt(self, name: str) -> str:
        return (self.prompts_dir / name).read_text(encoding="utf-8")

    def _paper_json(self, paper: PaperMetadata, **extra: object) -> str:
        payload = paper.model_dump(mode="json", exclude={"source_metadata"})
        payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)

    def _classify(self, paper: PaperMetadata, journal_name: str, model: str) -> DeepSeekResponse:
        assert self.client is not None
        user = self._read_prompt("relevance_user.txt").format(
            paper_json=self._paper_json(paper, journal=journal_name)
        )
        return self.client.request_json(
            model=model,
            system_prompt=self._read_prompt("relevance_system.txt"),
            user_prompt=user,
            schema=AIScreeningResult,
        )

    def _summarize(
        self, paper: PaperMetadata, classification: AIScreeningResult, model: str
    ) -> DeepSeekResponse:
        assert self.client is not None
        user = self._read_prompt("summary_user.txt").format(
            paper_json=self._paper_json(
                paper, classification=classification.model_dump(mode="json")
            )
        )
        return self.client.request_json(
            model=model,
            system_prompt=self._read_prompt("summary_system.txt"),
            user_prompt=user,
            schema=SummaryResult,
        )

    @staticmethod
    def _count(stats: ScreeningStats, response: DeepSeekResponse) -> None:
        stats.ai_calls += 1
        stats.input_tokens += response.input_tokens
        stats.output_tokens += response.output_tokens

    def _save(
        self,
        paper_id: int,
        rule: RuleResult,
        model: str,
        decision: str,
        confidence: float | None,
        reason: str,
    ) -> None:
        self._write_screening(
            paper_id=paper_id,
            rule=rule,
            model=model,
            decision=decision,
            confidence=confidence,
            reason=reason,
        )

    def _save_result(
        self,
        paper_id: int,
        rule: RuleResult,
        model: str,
        result: AIScreeningResult,
        summary: SummaryResult | None,
    ) -> None:
        missing = list(
            dict.fromkeys(
                [*result.missing_information, *(summary.missing_information if summary else [])]
            )
        )
        self._write_screening(
            paper_id=paper_id,
            rule=rule,
            model=model,
            decision=result.decision,
            confidence=result.confidence,
            reason=result.reason,
            topics=result.topics,
            education_level=result.education_level,
            country_or_region=result.country_or_region,
            method=result.method,
            research_question=(summary.research_question if summary else None)
            or result.research_question,
            data_and_sample=summary.data_and_sample if summary else None,
            identification_strategy=summary.identification_strategy if summary else None,
            main_result=(summary.main_result if summary else None) or result.main_result,
            heterogeneity=summary.heterogeneity if summary else None,
            policy_implication=summary.policy_implication if summary else None,
            chinese_title=summary.chinese_title if summary else None,
            chinese_summary=summary.chinese_summary if summary else None,
            missing=missing,
        )

    def _write_screening(self, **values: object) -> None:
        columns = [
            "paper_id",
            "rule_score",
            "rule_hits_json",
            "ai_decision",
            "ai_confidence",
            "ai_reason",
            "topics_json",
            "education_level",
            "country_or_region",
            "method",
            "research_question",
            "data_and_sample",
            "identification_strategy",
            "main_result",
            "heterogeneity",
            "policy_implication",
            "chinese_title",
            "chinese_summary",
            "missing_information_json",
            "model_name",
            "prompt_version",
            "screened_at",
        ]
        rule = values["rule"]
        assert isinstance(rule, RuleResult)
        row = [
            values["paper_id"],
            rule.score,
            json.dumps(
                {"positive": rule.positive_hits, "risk": rule.risk_hits}, ensure_ascii=False
            ),
            values["decision"],
            values.get("confidence"),
            values.get("reason"),
            json.dumps(values.get("topics", []), ensure_ascii=False),
            values.get("education_level"),
            values.get("country_or_region"),
            values.get("method"),
            values.get("research_question"),
            values.get("data_and_sample"),
            values.get("identification_strategy"),
            values.get("main_result"),
            values.get("heterogeneity"),
            values.get("policy_implication"),
            values.get("chinese_title"),
            values.get("chinese_summary"),
            json.dumps(values.get("missing", []), ensure_ascii=False),
            values["model"],
            PROMPT_VERSION,
            utcnow(),
        ]
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(
            f"{column}=excluded.{column}"
            for column in columns
            if column not in {"paper_id", "model_name", "prompt_version"}
        )
        with self.database.connect() as connection:
            connection.execute(
                f"INSERT INTO screenings ({','.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(paper_id,model_name,prompt_version) DO UPDATE SET {updates}",
                row,
            )

    def _record_usage(self, stats: ScreeningStats) -> None:
        if not stats.ai_calls:
            return
        with self.database.connect() as connection:
            row = connection.execute("SELECT id FROM sync_runs ORDER BY id DESC LIMIT 1").fetchone()
            if row:
                connection.execute(
                    """UPDATE sync_runs SET ai_calls=ai_calls+?,input_tokens=input_tokens+?,
                    output_tokens=output_tokens+? WHERE id=?""",
                    (stats.ai_calls, stats.input_tokens, stats.output_tokens, row["id"]),
                )


def preview_screening_candidates(
    database: Database, *, source: str, limit: int
) -> tuple[int, list[dict]]:
    if source not in {"awaiting_screening", "screening_failed"}:
        raise ValueError("Unsupported screening source")
    if not 1 <= limit <= 100:
        raise ValueError("Screening limit must be between 1 and 100")
    if source == "awaiting_screening":
        condition = "NOT EXISTS (SELECT 1 FROM screenings s WHERE s.paper_id=p.id)"
    else:
        condition = """'unprocessed'=(
            SELECT s.ai_decision FROM screenings s WHERE s.paper_id=p.id
            ORDER BY s.screened_at DESC,s.id DESC LIMIT 1
        )"""
    from_sql = f"""FROM papers p JOIN journals j ON j.id=p.journal_id
        WHERE {condition}"""
    with database.connect() as connection:
        total = int(connection.execute(f"SELECT COUNT(*) {from_sql}").fetchone()[0])
        rows = connection.execute(
            f"""SELECT p.id,p.title,j.name journal_name {from_sql}
            ORDER BY p.first_seen_at,p.id LIMIT ?""",
            (limit,),
        ).fetchall()
    return total, [dict(row) for row in rows]
