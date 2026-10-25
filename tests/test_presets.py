"""Unit tests for preset scenario datasets."""

from src.core.preset_data import get_all_presets, get_preset_by_id


def test_presets_exist() -> None:
    presets = get_all_presets()
    assert len(presets) >= 5


def test_presets_schema_validation() -> None:
    presets = get_all_presets()
    categories = {p.category for p in presets}
    assert "Coreference" in categories
    assert "Grammar & Syntax" in categories
    assert "Induction Circuit" in categories

    for p in presets:
        assert p.id
        assert p.title
        assert len(p.sentence.split()) >= 3
        assert len(p.key_pairs) >= 1
        assert len(p.explanation) > 20


def test_get_preset_by_id() -> None:
    preset = get_preset_by_id("winograd-tired")
    assert preset is not None
    assert "animal" in preset.sentence
    assert "tired" in preset.sentence

    missing = get_preset_by_id("non-existent-id")
    assert missing is None
