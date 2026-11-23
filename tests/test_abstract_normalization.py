from eduecon_radar.services.abstract_normalization import (
    content_sha256,
    normalize_abstract,
    validate_abstract_quality,
)


def test_jats_html_entities_and_hash_are_normalized():
    raw = (
        "<abstract><title>Abstract</title><p>"
        + "Education &amp; earnings evidence. " * 25
        + "</p></abstract>"
    )
    text = normalize_abstract(raw)
    assert "<" not in text and "&amp;" not in text
    assert content_sha256(text) == content_sha256(text)
    assert validate_abstract_quality(text).accepted


def test_quality_rejects_noise_short_non_english_and_duplicate():
    assert "too_short" in validate_abstract_quality("No abstract available").rejection_reasons
    chinese = "教育经济学研究" * 30
    assert "not_primarily_english" in validate_abstract_quality(chinese).rejection_reasons
    text = "Education outcomes and teacher quality are estimated with longitudinal data. " * 8
    digest = content_sha256(text.strip())
    assert (
        "duplicate_content"
        in validate_abstract_quality(text, duplicate_hashes={digest}).rejection_reasons
    )
