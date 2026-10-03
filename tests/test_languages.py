"""Tests for kugelaudio_open.languages — run with: pytest tests/test_languages.py -v"""

import pytest
from kugelaudio_open.languages import (
    LANGUAGES, DISPLAY_ORDER, DEFAULT_LANG,
    get, codes, validate, quality_warning,
    gradio_choices, parse_gradio_choice,
)


class TestLanguageData:
    def test_count(self):
        assert len(LANGUAGES) == 23

    def test_display_order_consistent(self):
        assert set(DISPLAY_ORDER) == set(LANGUAGES.keys())

    def test_high_tier(self):
        for c in ["en", "de", "fr", "es"]:
            assert LANGUAGES[c].tier == "high"

    def test_all_fields_populated(self):
        for code, lang in LANGUAGES.items():
            assert lang.code == code
            assert lang.name and lang.native_name and lang.flag
            assert lang.tier in ("high", "medium", "limited")

    def test_frozen(self):
        with pytest.raises(AttributeError):
            LANGUAGES["en"].name = "Nope"


class TestLookup:
    def test_valid(self):
        assert get("de").name == "German"

    def test_case_insensitive(self):
        assert get("DE") is not None

    def test_whitespace(self):
        assert get("  fr  ") is not None

    def test_invalid(self):
        assert get("xx") is None


class TestValidate:
    def test_ok(self):
        assert validate("en") == "en"
        assert validate("DE") == "de"

    def test_bad(self):
        with pytest.raises(ValueError, match="Unsupported"):
            validate("xx")


class TestQualityWarning:
    def test_high_none(self):
        assert quality_warning("en") is None

    def test_medium_none(self):
        assert quality_warning("it") is None

    def test_limited_warns(self):
        w = quality_warning("bg")
        assert w and "⚠️" in w


class TestGradio:
    def test_choices_count(self):
        assert len(gradio_choices()) == len(LANGUAGES)

    def test_limited_has_warning(self):
        bg = [c for c in gradio_choices() if "(bg)" in c][0]
        assert "⚠️" in bg

    def test_high_no_warning(self):
        en = [c for c in gradio_choices() if "(en)" in c][0]
        assert "⚠️" not in en

    def test_parse_roundtrip(self):
        for choice, expected in zip(gradio_choices(), codes()):
            assert parse_gradio_choice(choice) == expected

    def test_parse_fallback(self):
        assert parse_gradio_choice("garbage") == DEFAULT_LANG
