"""Unit tests for web application assets and integrity."""

import json
from pathlib import Path


def test_web_assets_exist() -> None:
    base = Path(__file__).resolve().parent.parent
    assert (base / "index.html").exists()
    assert (base / "style.css").exists()
    assert (base / "app.js").exists()
    assert (base / ".nojekyll").exists()
    assert (base / ".gitignore").exists()
    assert (base / "data" / "presets.json").exists()


def test_presets_json_validity() -> None:
    base = Path(__file__).resolve().parent.parent
    presets_file = base / "data" / "presets.json"
    assert presets_file.exists()

    with open(presets_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 5

    for item in data:
        assert "id" in item
        assert "sentence" in item
        assert "key_pairs" in item
        assert "explanation" in item


def test_html_contains_critical_components() -> None:
    base = Path(__file__).resolve().parent.parent
    html = (base / "index.html").read_text(encoding="utf-8")

    assert "matrix-grid" in html
    assert "attention-flow-svg" in html
    assert "math-inspector" in html
    assert "temp-slider" in html
    assert "btn-causal" in html
    assert "btn-bidirectional" in html
