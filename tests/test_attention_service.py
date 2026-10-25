"""Unit tests for AttentionService and core attention math."""

import math

from src.core.attention_service import (
    AttentionService,
    scaled_dot_product_attention,
    softmax,
    tokenize_sentence,
)


def test_tokenize_sentence() -> None:
    tokens = tokenize_sentence("The animal didn't cross the street.")
    assert len(tokens) >= 6
    assert tokens[0] == "The"
    assert "animal" in tokens
    assert "street" in tokens


def test_softmax_properties() -> None:
    scores = [1.0, 2.0, 3.0, 4.0]
    weights = softmax(scores, temperature=1.0)
    assert len(weights) == 4
    assert math.isclose(sum(weights), 1.0, rel_tol=1e-5)
    # Higher score must have higher probability
    assert weights[3] > weights[2] > weights[1] > weights[0]


def test_softmax_temperature_effect() -> None:
    scores = [1.0, 2.0, 5.0]
    hot = softmax(scores, temperature=5.0)  # Diffuse / uniform
    cold = softmax(scores, temperature=0.2)  # Sharp / peaked

    # Hot distribution should have higher entropy (closer values)
    assert hot[2] < cold[2]
    assert hot[0] > cold[0]
    assert math.isclose(cold[2], 1.0, abs_tol=0.01)


def test_scaled_dot_product_basic() -> None:
    # 2 tokens, d_k = 2
    Q = [[1.0, 0.0], [0.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[10.0, 20.0], [30.0, 40.0]]
    d_k = 2

    raw_scores, weights, outputs = scaled_dot_product_attention(
        Q, K, V, d_k=d_k, is_causal=False, temperature=1.0
    )

    # Check shapes
    assert len(weights) == 2
    assert len(weights[0]) == 2
    assert len(outputs) == 2
    assert len(outputs[0]) == 2

    # Row sums must be 1.0
    assert math.isclose(sum(weights[0]), 1.0, rel_tol=1e-5)
    assert math.isclose(sum(weights[1]), 1.0, rel_tol=1e-5)

    # Q0 matches K0 exactly, so weights[0][0] > weights[0][1]
    assert weights[0][0] > weights[0][1]


def test_causal_masking() -> None:
    # 3 tokens
    Q = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    K = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    V = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    d_k = 2

    _, weights, _ = scaled_dot_product_attention(Q, K, V, d_k=d_k, is_causal=True, temperature=1.0)

    # Token 0 can only attend to Token 0
    assert math.isclose(weights[0][0], 1.0, rel_tol=1e-5)
    assert weights[0][1] == 0.0
    assert weights[0][2] == 0.0

    # Token 1 can only attend to Token 0 and Token 1
    assert math.isclose(sum(weights[1][:2]), 1.0, rel_tol=1e-5)
    assert weights[1][2] == 0.0

    # Token 2 can attend to all 3
    assert math.isclose(sum(weights[2]), 1.0, rel_tol=1e-5)


def test_attention_service_multi_head() -> None:
    service = AttentionService()
    result = service.compute_attention(
        sentence="The animal didn't cross the street because it was too tired",
        is_causal=False,
        temperature=1.0,
        head_count=4,
    )

    assert result.token_count > 5
    assert len(result.heads) == 4
    assert len(result.composite_matrix) == result.token_count
    assert len(result.composite_matrix[0]) == result.token_count

    # Check composite matrix rows sum to 1.0
    for row in result.composite_matrix:
        assert math.isclose(sum(row), 1.0, rel_tol=1e-4)


def test_induction_head_copy_pattern() -> None:
    service = AttentionService()
    # Sentence with repeated pattern: Harry Potter Harry Potter
    result = service.compute_attention(
        sentence="Harry Potter Harry Potter",
        is_causal=True,
        temperature=1.0,
    )
    # Head 2 is induction head
    induction_head = result.heads[2]
    assert induction_head.specialization == "Induction / Pattern Copying"
