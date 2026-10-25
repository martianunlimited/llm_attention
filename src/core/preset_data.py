"""Preset scenarios for Multi-Head Self-Attention educational lab."""

import json
from pathlib import Path

from src.config import settings
from src.schemas import KeyAttentionPair, PresetScenario

PRESETS: list[PresetScenario] = [
    PresetScenario(
        id="winograd-tired",
        title="Winograd Schema: The Tired Animal",
        category="Coreference",
        description="Ambiguous pronoun resolution driven by predicate adjective 'tired'.",
        sentence="The animal didn't cross the street because it was too tired",
        default_head=1,
        key_pairs=[
            KeyAttentionPair(
                source_token="it",
                source_idx=7,
                target_token="animal",
                target_idx=1,
                expected_strength="high",
                reason=(
                    "'Tired' is an animate predicate attaching to living beings "
                    "('animal'), not infrastructure ('street')."
                ),
            ),
            KeyAttentionPair(
                source_token="tired",
                source_idx=10,
                target_token="animal",
                target_idx=1,
                expected_strength="medium",
                reason="Adjective 'tired' qualifies the antecedent entity.",
            ),
        ],
        explanation=(
            "In the Coreference Head (Head 1), the query vector of 'it' has a high dot "
            "product with the key vector of 'animal'. This demonstrates how self-attention "
            "resolves ambiguous pronouns by integrating contextual cues from the entire sentence."
        ),
    ),
    PresetScenario(
        id="winograd-wide",
        title="Winograd Schema: The Wide Street",
        category="Coreference",
        description=(
            "Swapping 'tired' with 'wide' causes the exact same pronoun 'it' to shift "
            "its attention to 'street'."
        ),
        sentence="The animal didn't cross the street because it was too wide",
        default_head=1,
        key_pairs=[
            KeyAttentionPair(
                source_token="it",
                source_idx=7,
                target_token="street",
                target_idx=5,
                expected_strength="high",
                reason=(
                    "'Wide' is a spatial measurement adjective applying to geographic pathways "
                    "('street'), not animals."
                ),
            )
        ],
        explanation=(
            "Notice how changing just one word at the end of the sentence completely "
            "alters the self-attention pattern for 'it'. Attention is dynamic and routes "
            "information based on semantic compatibility."
        ),
    ),
    PresetScenario(
        id="subject-verb",
        title="Long-Distance Agreement: Keys to the Cabinet",
        category="Grammar & Syntax",
        description=(
            "The plural verb 'are' must attend over the singular distractor 'cabinet' "
            "to match plural 'keys'."
        ),
        sentence="The keys to the cabinet are on the wooden table",
        default_head=0,
        key_pairs=[
            KeyAttentionPair(
                source_token="are",
                source_idx=5,
                target_token="keys",
                target_idx=1,
                expected_strength="high",
                reason=(
                    "Plural auxiliary 'are' must agree in number with subject 'keys', "
                    "ignoring singular noun 'cabinet'."
                ),
            )
        ],
        explanation=(
            "In the Syntax Head (Head 0), 'are' learns to skip the intervening prepositional "
            "phrase ('to the cabinet') and form a strong attention bridge directly back to 'keys'."
        ),
    ),
    PresetScenario(
        id="induction-pattern",
        title="Anthropic Induction Circuit: Pattern Copying",
        category="Induction Circuit",
        description="How Transformers implement in-context learning: [A][B] ... [A] -> [B].",
        sentence="Harry Potter and Harry Potter",
        default_head=2,
        key_pairs=[
            KeyAttentionPair(
                source_token="Harry",
                source_idx=3,
                target_token="Potter",
                target_idx=1,
                expected_strength="high",
                reason=(
                    "The induction head attends from the second 'Harry' back to 'Potter' "
                    "(the token that followed the previous 'Harry') to predict continuation."
                ),
            )
        ],
        explanation=(
            "Discovered by Anthropic in 'In-context Learning and Induction Heads' (2022). "
            "When a token reappears, the induction head looks at earlier instances of that "
            "token and shifts attention to the subsequent token, enabling few-shot pattern copying."
        ),
    ),
    PresetScenario(
        id="polysemy-disambiguation",
        title="Polysemy Disambiguation: River Bank vs Financial Bank",
        category="Contextual Disambiguation",
        description="Disambiguating homonyms through self-attention context exchange.",
        sentence="The river bank was muddy after the heavy rain",
        default_head=1,
        key_pairs=[
            KeyAttentionPair(
                source_token="bank",
                source_idx=2,
                target_token="river",
                target_idx=1,
                expected_strength="high",
                reason=(
                    "'river' injects aquatic features into 'bank', removing financial ambiguity."
                ),
            ),
            KeyAttentionPair(
                source_token="bank",
                source_idx=2,
                target_token="muddy",
                target_idx=4,
                expected_strength="high",
                reason="'muddy' reinforces the geological soil interpretation.",
            ),
        ],
        explanation=(
            "Before self-attention, the embedding for 'bank' is generic. Once attention weights "
            "are computed, the value vectors of 'river' and 'muddy' are added into 'bank', "
            "enriching its hidden state representation so downstream layers know it is "
            "an embankment."
        ),
    ),
    PresetScenario(
        id="code-variable-scope",
        title="Code & Logic: Variable Scoping in Functions",
        category="Code & Logic",
        description="Tracking identifier references inside programming code syntax.",
        sentence="def calculate_total ( items ) : return sum ( items )",
        default_head=0,
        key_pairs=[
            KeyAttentionPair(
                source_token="items",
                source_idx=9,
                target_token="items",
                target_idx=3,
                expected_strength="high",
                reason="Return expression identifier attends to function parameter declaration.",
            )
        ],
        explanation=(
            "In code-specialized LLMs, attention heads specialize in linking variable uses "
            "back to their definitions across scope boundaries."
        ),
    ),
]


def get_all_presets() -> list[PresetScenario]:
    """Retrieve all preset scenarios."""
    return PRESETS


def get_preset_by_id(preset_id: str) -> PresetScenario | None:
    """Retrieve a single preset by ID."""
    for p in PRESETS:
        if p.id == preset_id:
            return p
    return None


def export_presets_json(output_path: Path | None = None) -> Path:
    """Export presets to JSON for web consumption."""
    target_path = output_path or settings.presets_file
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in PRESETS], f, indent=2)
    return target_path
