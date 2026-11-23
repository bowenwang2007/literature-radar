"""Phase 6A acceptance tests: product workflow reorientation."""

# Some assert expressions intentionally exceed 100 chars for readability.
# ruff: noqa: E501

from eduecon_radar.services.prototype_workflow import _compute_priority_label


class TestComputePriorityLabel:
    def test_core_edu_economics(self):
        assert _compute_priority_label("include", 0.9, "strong_positive", False) == "核心教育经济学"

    def test_core_edu_economics_with_moderate_evidence(self):
        assert _compute_priority_label("include", 0.8, "moderate_positive", False) == "核心教育经济学"

    def test_highly_relevant(self):
        assert _compute_priority_label("include", 0.85, "little_evidence", False) == "高度相关"

    def test_highly_relevant_at_threshold(self):
        assert _compute_priority_label("include", 0.7, "weak_or_mixed", False) == "高度相关"

    def test_suggest_manual_check_low_confidence_include(self):
        assert _compute_priority_label("include", 0.5, "little_evidence", False) == "建议人工检查"

    def test_suggest_manual_check_borderline(self):
        assert _compute_priority_label("borderline", 0.6, "strong_positive", False) == "建议人工检查"

    def test_suggest_manual_check_conflict(self):
        assert _compute_priority_label("exclude", 0.3, "moderate_positive", True) == "建议人工检查"

    def test_low_relevance(self):
        assert _compute_priority_label("exclude", 0.1, "little_evidence", False) == "低相关"

    def test_no_fake_score(self):
        """priority_label is a categorical tag, not a numeric score."""
        label = _compute_priority_label("include", 0.9, "strong_positive", False)
        assert label not in ("90", "100", "85.0", "0.9")
        assert isinstance(label, str)

    def test_not_hardcoded_mapping_from_decision(self):
        """Same decision but different confidence/evidence produces different labels."""
        core = _compute_priority_label("include", 0.9, "strong_positive", False)
        manual = _compute_priority_label("include", 0.5, "little_evidence", False)
        assert core != manual


class TestSchemaGate:
    """_missing_schema_layers checks all four required schemas."""

    def test_import_schema_helpers(self):
        from eduecon_radar.domain.semantic_storage import semantic_schema_ready
        from eduecon_radar.domain.storage import domain_evidence_schema_ready
        from eduecon_radar.services.migrations import (
            translation_schema_ready,
            workflow_schema_ready,
        )
        assert callable(translation_schema_ready)
        assert callable(workflow_schema_ready)
        assert callable(domain_evidence_schema_ready)
        assert callable(semantic_schema_ready)

class TestPhase6AConfig:
    def test_default_ui_version_is_v05(self):
        import tempfile
        from pathlib import Path

        from eduecon_radar.config import Settings
        s = Settings.load(Path(tempfile.mkdtemp()))
        assert s.ui_version == "v05"

    def test_explicit_v04_still_works(self, monkeypatch):
        import tempfile
        from pathlib import Path

        from eduecon_radar.config import Settings
        monkeypatch.setenv("EDUECON_UI_VERSION", "v04")
        s = Settings.load(Path(tempfile.mkdtemp()))
        assert s.ui_version == "v04"

    def test_settings_rejects_unknown_version(self, monkeypatch):
        import tempfile
        from pathlib import Path

        from eduecon_radar.config import Settings
        monkeypatch.setenv("EDUECON_UI_VERSION", "v99")
        import pytest
        with pytest.raises(ValueError, match="EDUECON_UI_VERSION"):
            Settings.load(Path(tempfile.mkdtemp()))

    def test_env_example_has_no_retired_ui_mode_switch(self):
        from pathlib import Path
        env_example = Path(__file__).resolve().parents[1] / ".env.example"
        content = env_example.read_text(encoding="utf-8")
        assert "EDUECON_UI_VERSION" not in content
        assert "EDUERADAR_PERSONAL_MODE" not in content

    def test_run_web_ps1_does_not_force_v04(self):
        from pathlib import Path
        run_web = Path(__file__).resolve().parents[1] / "run_web.ps1"
        content = run_web.read_text(encoding="utf-8")
        assert "EDUECON_UI_VERSION" not in content or "v04" not in content


class TestDecisionDetailFields:
    """_decision_detail adds priority_label and related fields to its return dict."""

    def test_decision_detail_path_exists(self):
        import inspect

        from eduecon_radar.services.prototype_workflow import _decision_detail
        sig = inspect.signature(_decision_detail)
        assert list(sig.parameters.keys()) == ["database", "paper_id", "evidence", "analysis", "failure"]
        source = inspect.getsource(_decision_detail)
        assert "priority_label" in source
        assert "semantic_score" in source
        assert "score_is_probability" in source
        assert "uncertainty_label" in source
        assert "missed_priority_label" in source


class TestMentorExportSafety:
    def test_mentor_source_code_excludes_ai_fields(self):
        import inspect

        from eduecon_radar.services import exporting
        source = inspect.getsource(exporting.mentor_records)
        source2 = inspect.getsource(exporting.export_mentor_markdown)
        source3 = inspect.getsource(exporting.export_mentor_csv_text)
        combined = source + source2 + source3
        ai_terms = {"ai_decision", "ai_confidence", "ai_reason",
                    "semantic_score", "priority_label",
                    "priority_tier", "priority_score",
                    "chinese_summary", "research_question",
                    "data_and_sample", "identification_strategy",
                    "main_result", "heterogeneity", "policy_implication"}
        for term in ai_terms:
            assert term not in combined, f"AI/personal-only field '{term}' found in mentor export code"


class TestWebUiPagesImportability:
    """Verify shipped page modules import without network or external services."""

    def test_personal_mode_imports(self):
        import eduecon_radar.ui.views.personal_mode as m
        assert callable(m.render)

    def test_paper_library_imports(self):
        import eduecon_radar.ui.views.paper_library as m
        assert callable(m.render)

    def test_app_py_imports(self):
        import eduecon_radar.ui.app as m
        assert hasattr(m, "main")
        assert callable(m.main)
