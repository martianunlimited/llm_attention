"""Core Attention Service: Multi-Head Self-Attention mathematical calculations."""

import math
import re

from src.config import settings
from src.schemas import (
    AttentionHead,
    MultiHeadAttentionResult,
)


def tokenize_sentence(text: str) -> list[str]:
    """Tokenize a sentence into words and punctuation marks."""
    # Split by whitespace, punctuation, keeping word contractions together if needed
    cleaned = text.strip()
    if not cleaned:
        return []
    # Match words, contractions (didn't, isn't), or single punctuation marks
    pattern = r"\b[\w']+\b|[^\w\s]"
    tokens = re.findall(pattern, cleaned)
    return tokens if tokens else cleaned.split()


def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax with temperature scaling."""
    t = max(0.01, temperature)

    # Check for -inf or masked values
    valid_scores = [s for s in scores if s > -1e8]
    if not valid_scores:
        # All masked
        return [0.0] * len(scores)

    max_val = max(valid_scores)
    exp_scores: list[float] = []

    for s in scores:
        if s <= -1e8:
            exp_scores.append(0.0)
        else:
            exp_scores.append(math.exp((s - max_val) / t))

    total = sum(exp_scores)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)

    return [e / total for e in exp_scores]


def scaled_dot_product_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    d_k: int,
    is_causal: bool = False,
    temperature: float = 1.0,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Compute Scaled Dot-Product Attention: softmax(QK^T / sqrt(d_k)) * V."""
    n_queries = len(Q)
    n_keys = len(K)
    scale = math.sqrt(max(1, d_k))

    raw_scores: list[list[float]] = []
    attention_weights: list[list[float]] = []
    outputs: list[list[float]] = []

    for i in range(n_queries):
        row_raw: list[float] = []
        row_scaled: list[float] = []

        for j in range(n_keys):
            if is_causal and j > i:
                row_raw.append(-1e9)
                row_scaled.append(-1e9)
            else:
                dot = sum(Q[i][d] * K[j][d] for d in range(d_k))
                row_raw.append(dot)
                row_scaled.append(dot / scale)

        weights = softmax(row_scaled, temperature=temperature)
        raw_scores.append(row_raw)
        attention_weights.append(weights)

        # Output: sum_j weights[j] * V[j]
        v_dim = len(V[0]) if V else d_k
        out_vec = [0.0] * v_dim
        for j in range(n_keys):
            w = weights[j]
            for d in range(v_dim):
                out_vec[d] += w * V[j][d]
        outputs.append(out_vec)

    return raw_scores, attention_weights, outputs


class AttentionService:
    """Service to generate multi-head self-attention for sentences."""

    def __init__(self) -> None:
        self.d_k = settings.default_d_k
        self.head_count = settings.default_head_count

    def _generate_head_vectors(
        self, tokens: list[str], head_idx: int, d_k: int
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        """Generate projection vectors Q, K, V tailored to each head's functional role."""
        n = len(tokens)
        Q: list[list[float]] = []
        K: list[list[float]] = []
        V: list[list[float]] = []

        tokens_lower = [t.lower() for t in tokens]
        has_tired = "tired" in tokens_lower
        has_wide = "wide" in tokens_lower

        for i, tok in enumerate(tokens_lower):
            # Base pseudo-random vector derived from token string
            base_seed = sum(ord(c) * (pos + 1) for pos, c in enumerate(tok))
            q_vec = [math.sin(base_seed + d * 1.31) for d in range(d_k)]
            k_vec = [math.cos(base_seed + d * 1.31) for d in range(d_k)]
            v_vec = [math.sin(base_seed * 0.5 + d * 0.77) for d in range(d_k)]

            # Specialize vectors according to head function
            if head_idx == 0:
                # HEAD 0: Syntactic & Local Context Head
                # Attends to adjacent tokens and structural anchors (verbs <-> subjects)
                q_vec[0] = float(i) * 0.5
                k_vec[0] = float(i) * 0.5
                # Boost verb-subject coupling
                if tok in ("are", "is", "was", "were"):
                    q_vec[1] = 3.0
                if tok in ("keys", "animal", "street", "table", "cabinet"):
                    k_vec[1] = 2.5 if tok == "keys" else 0.5

            elif head_idx == 1:
                # HEAD 1: Coreference & Semantic Resolution Head
                # Pronouns seek matching entities
                if tok in ("it", "they", "he", "she", "this", "that"):
                    q_vec[0] = 4.0
                    q_vec[1] = 4.0 if has_tired else (-4.0 if has_wide else 1.0)

                # Entity keys
                if tok == "animal":
                    k_vec[0] = 3.5
                    k_vec[1] = 4.0  # matches 'it' when has_tired is true
                elif tok == "street":
                    k_vec[0] = 3.5
                    k_vec[1] = -4.0  # matches 'it' when has_wide is true

                # Polysemy 'bank'
                if tok == "bank":
                    q_vec[2] = 3.0
                if tok in ("river", "muddy", "water", "rain"):
                    k_vec[2] = 3.5

            elif head_idx == 2:
                # HEAD 2: Anthropic Induction & Pattern Copying Head
                # When token occurs a second time, attend to the token after previous occurrence
                # Find if this token occurred earlier
                prev_indices = [idx for idx, t in enumerate(tokens_lower[:i]) if t == tok]
                if prev_indices:
                    # Point query strongly to match keys of subsequent tokens
                    q_vec[3] = 5.0
                    prev_target = prev_indices[-1] + 1
                    if prev_target < n:
                        # Artificially align with the target token's key
                        target_tok = tokens_lower[prev_target]
                        target_seed = sum(ord(c) * (p + 1) for p, c in enumerate(target_tok))
                        q_vec = [math.cos(target_seed + d * 1.31) * 3.0 for d in range(d_k)]
                else:
                    q_vec[3] = -1.0

            elif head_idx == 3:
                # HEAD 3: Global Anchor Head (Attends to [CLS] / initial root)
                q_vec[4 % d_k] = 3.0
                if i == 0:
                    k_vec[4 % d_k] = 5.0
                else:
                    k_vec[4 % d_k] = 0.5

            # Normalize vectors to reasonable magnitude
            q_norm = math.sqrt(sum(x * x for x in q_vec)) or 1.0
            k_norm = math.sqrt(sum(x * x for x in k_vec)) or 1.0
            v_norm = math.sqrt(sum(x * x for x in v_vec)) or 1.0

            Q.append([x / q_norm * 2.0 for x in q_vec])
            K.append([x / k_norm * 2.0 for x in k_vec])
            V.append([x / v_norm * 1.5 for x in v_vec])

        return Q, K, V

    def compute_attention(
        self,
        sentence: str,
        is_causal: bool = False,
        temperature: float = 1.0,
        head_count: int = 4,
    ) -> MultiHeadAttentionResult:
        """Compute full multi-head self-attention for a sentence."""
        tokens = tokenize_sentence(sentence)
        n = len(tokens)
        d_k = self.d_k

        if n == 0:
            return MultiHeadAttentionResult(
                sentence=sentence,
                tokens=[],
                token_count=0,
                d_k=d_k,
                head_count=head_count,
                is_causal=is_causal,
                temperature=temperature,
                heads=[],
                composite_matrix=[],
            )

        heads_meta = [
            (
                "Head 0: Syntactic & Local Context",
                "Syntax",
                "Attends to adjacent tokens, prepositions, and grammatical subject-verb linkages.",
            ),
            (
                "Head 1: Coreference & Semantic Resolution",
                "Coreference",
                (
                    "Resolves pronouns (it, they) to their semantic antecedents via "
                    "context adjectives."
                ),
            ),
            (
                "Head 2: Induction Circuit (Pattern Copying)",
                "Induction / Pattern Copying",
                (
                    "Implements in-context learning: looks back at prior occurrences "
                    "to copy succeeding tokens."
                ),
            ),
            (
                "Head 3: Global Anchor & Prefix Head",
                "Global Anchor",
                "Anchors information to sentence root or prompt origin tokens.",
            ),
        ]

        computed_heads: list[AttentionHead] = []

        # Initialize composite attention matrix
        composite = [[0.0] * n for _ in range(n)]

        for h in range(head_count):
            meta_name, meta_spec, meta_desc = (
                heads_meta[h]
                if h < len(heads_meta)
                else (f"Head {h}", "General", "Multi-head projection")
            )
            Q, K, V = self._generate_head_vectors(tokens, h, d_k)
            raw_scores, weights, outputs = scaled_dot_product_attention(
                Q, K, V, d_k=d_k, is_causal=is_causal, temperature=temperature
            )

            # Accumulate into composite
            for i in range(n):
                for j in range(n):
                    composite[i][j] += weights[i][j] / head_count

            head = AttentionHead(
                id=h,
                name=meta_name,
                specialization=meta_spec,
                description=meta_desc,
                d_k=d_k,
                queries=Q,
                keys=K,
                values=V,
                raw_scores=raw_scores,
                attention_matrix=weights,
                output_vectors=outputs,
            )
            computed_heads.append(head)

        return MultiHeadAttentionResult(
            sentence=sentence,
            tokens=tokens,
            token_count=n,
            d_k=d_k,
            head_count=head_count,
            is_causal=is_causal,
            temperature=temperature,
            heads=computed_heads,
            composite_matrix=composite,
        )
