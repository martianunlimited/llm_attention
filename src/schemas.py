"""Pydantic v2 schemas for Multi-Head Self-Attention Lab."""

from pydantic import BaseModel, Field


class TokenVector(BaseModel):
    """Represents a token and its vector representation."""

    index: int
    text: str
    vector: list[float] = Field(default_factory=list)


class AttentionCell(BaseModel):
    """Calculated attention values between query and key tokens."""

    query_idx: int
    key_idx: int
    query_token: str
    key_token: str
    raw_dot_product: float
    scaled_score: float
    weight: float
    is_masked: bool = False


class AttentionHead(BaseModel):
    """Data for an individual attention head."""

    id: int
    name: str
    specialization: str
    description: str
    d_k: int
    queries: list[list[float]] = Field(default_factory=list)
    keys: list[list[float]] = Field(default_factory=list)
    values: list[list[float]] = Field(default_factory=list)
    raw_scores: list[list[float]] = Field(default_factory=list)
    attention_matrix: list[list[float]] = Field(default_factory=list)
    output_vectors: list[list[float]] = Field(default_factory=list)


class KeyAttentionPair(BaseModel):
    """Key attention pairing to highlight in educational presets."""

    source_token: str
    source_idx: int
    target_token: str
    target_idx: int
    expected_strength: str  # "high", "medium", "low"
    reason: str


class PresetScenario(BaseModel):
    """Educational scenario demonstrating a linguistic or computational pattern."""

    id: str
    title: str
    category: str
    description: str
    sentence: str
    default_head: int = 0
    key_pairs: list[KeyAttentionPair] = Field(default_factory=list)
    explanation: str


class MultiHeadAttentionResult(BaseModel):
    """Complete multi-head attention result across all tokens and heads."""

    sentence: str
    tokens: list[str]
    token_count: int
    d_k: int
    head_count: int
    is_causal: bool
    temperature: float
    heads: list[AttentionHead]
    composite_matrix: list[list[float]]
