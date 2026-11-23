from pathlib import Path

from eduecon_radar.db import Database
from eduecon_radar.schemas import AIScreeningResult, JournalConfig, PaperMetadata, SummaryResult
from eduecon_radar.services.ai_screening import ScreeningService
from eduecon_radar.services.deepseek import DeepSeekResponse


class FakeDeepSeek:
    def __init__(self):
        self.calls = 0

    def request_json(self, *, model, system_prompt, user_prompt, schema):
        self.calls += 1
        if schema is AIScreeningResult:
            data = AIScreeningResult(
                decision="include",
                confidence=0.9,
                reason="Education is the core treatment and outcome domain.",
                topics=["教师质量"],
                research_question="Do teachers improve scores?",
            )
        else:
            data = SummaryResult(
                chinese_title="教师质量与学生成绩",
                research_question="教师是否改善学生成绩？",
                chinese_summary="本文研究教师质量与学生成绩之间的关系。",
            )
        return DeepSeekResponse(data=data, model_name=model, input_tokens=10, output_tokens=5)


def seeded_db(tmp_path):
    db = Database(tmp_path / "radar.db")
    db.init_schema()
    journal_id = db.upsert_journal(JournalConfig(name="Journal", short_name="J", issn="0000-0000"))
    db.upsert_paper(
        PaperMetadata(
            doi="10.1/teacher",
            title="Teacher quality and student scores",
            abstract="We study teacher quality in schools.",
        ),
        journal_id,
    )
    return db


def test_screening_is_cached_by_prompt_and_model(tmp_path):
    db = seeded_db(tmp_path)
    fake = FakeDeepSeek()
    service = ScreeningService(db, Path("prompts"), fake, "fast", "strong")
    first = service.run()
    second = service.run()
    assert first.screened == 1
    assert first.ai_calls == 2
    assert second.considered == 0
    assert fake.calls == 2
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM screenings").fetchone()
        assert row["ai_decision"] == "include"
        assert row["chinese_title"] == "教师质量与学生成绩"


def test_no_ai_retains_record_as_unprocessed(tmp_path):
    db = seeded_db(tmp_path)
    service = ScreeningService(db, Path("prompts"), None, "fast", "strong")
    stats = service.run(no_ai=True)
    assert stats.unprocessed == 1
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM screenings").fetchone()
        assert row["ai_decision"] == "unprocessed"
