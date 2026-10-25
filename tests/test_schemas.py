"""Unit tests for Pydantic v2 schemas."""

import pytest
from pydantic import ValidationError

from src.schemas import (
    AttentionCell,
    AttentionHead,
    KeyAttentionPair,
    PresetScenario,
    TokenVector,
)


def test_token_vector_schema() -> None:
    tv = TokenVector(index=0, text="animal", vector=[0.1, -0.2, 0.5])
    assert tv.index == 0
    assert tv.text == "animal"
    assert len(tv.vector) == 3


def test_attention_cell_schema() -> None:
    cell = AttentionCell(
        query_idx=0,
        key_idx=1,
        query_token="it",
        key_token="tired",
        raw_dot_product=2.5,
        scaled_score=0.884,
        weight=0.74,
        is_masked=False,
    )
    assert cell.query_token == "it"
    assert cell.weight == 0.74
    assert not cell.is_masked


def test_attention_head_schema() -> None:
    head = AttentionHead(
        id=0,
        name="Head 1",
        specialization="Coreference",
        description="Tracks pronoun references",
        d_k=4,
        queries=[[1.0, 0.0, 0.0, 0.0]],
        keys=[[1.0, 0.0, 0.0, 0.0]],
        values=[[0.5, 0.5, 0.0, 0.0]],
        raw_scores=[[1.0]],
        attention_matrix=[[1.0]],
        output_vectors=[[0.5, 0.5, 0.0, 0.0]],
    )
    assert head.id == 0
    assert head.d_k == 4
    assert len(head.attention_matrix) == 1


def test_key_attention_pair() -> None:
    pair = KeyAttentionPair(
        source_token="it",
        source_idx=7,
        target_token="animal",
        target_idx=1,
        expected_strength="high",
        reason="Pronoun 'it' refers to 'animal' because animal was tired",
    )
    assert pair.source_token == "it"
    assert pair.target_token == "animal"


def test_preset_scenario() -> None:
    preset = PresetScenario(
        id="winograd-tired",
        title="Winograd: The Tired Animal",
        category="Coreference",
        description="Testing ambiguous pronoun resolution",
        sentence="The animal didn't cross the street because it was too tired",
        default_head=1,
        key_pairs=[
            KeyAttentionPair(
                source_token="it",
                source_idx=7,
                target_token="animal",
                target_idx=1,
                expected_strength="high",
                reason="Tiredness resolves 'it' to 'animal'",
            )
        ],
        explanation="The coreference head links 'it' with high attention weight to 'animal'.",
    )
    assert preset.id == "winograd-tired"
    assert len(preset.key_pairs) == 1


def test_invalid_schema_rejection() -> None:
    with pytest.raises(ValidationError):
        # Missing required fields
        PresetScenario.model_validate({"id": "bad-scenario"})
