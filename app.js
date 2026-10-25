/**
 * LLM Multi-Head Self-Attention Lab - Interactive Client-Side Engine
 * Visualizing Q, K, V Projections, Scaled Dot Products, Softmax, Causal Masking,
 * Winograd Coreference, and Anthropic Induction Circuits.
 */

// Fallback presets if fetched via file:// protocol
const FALLBACK_PRESETS = [
  {
    "id": "winograd-tired",
    "title": "Winograd Schema: The Tired Animal",
    "category": "Coreference",
    "description": "Ambiguous pronoun resolution driven by predicate adjective 'tired'.",
    "sentence": "The animal didn't cross the street because it was too tired",
    "default_head": 1,
    "key_pairs": [
      {
        "source_token": "it",
        "source_idx": 7,
        "target_token": "animal",
        "target_idx": 1,
        "expected_strength": "high",
        "reason": "'Tired' is an animate predicate attaching to living beings ('animal'), not infrastructure ('street')."
      },
      {
        "source_token": "tired",
        "source_idx": 10,
        "target_token": "animal",
        "target_idx": 1,
        "expected_strength": "medium",
        "reason": "Adjective 'tired' qualifies the antecedent entity."
      }
    ],
    "explanation": "In the Coreference Head (Head 1), the query vector of 'it' has a high dot product with the key vector of 'animal'. This demonstrates how self-attention resolves ambiguous pronouns by integrating contextual cues from the entire sentence."
  },
  {
    "id": "winograd-wide",
    "title": "Winograd Schema: The Wide Street",
    "category": "Coreference",
    "description": "Swapping 'tired' with 'wide' causes the exact same pronoun 'it' to shift its attention to 'street'.",
    "sentence": "The animal didn't cross the street because it was too wide",
    "default_head": 1,
    "key_pairs": [
      {
        "source_token": "it",
        "source_idx": 7,
        "target_token": "street",
        "target_idx": 5,
        "expected_strength": "high",
        "reason": "'Wide' is a spatial measurement adjective applying to geographic pathways ('street'), not animals."
      }
    ],
    "explanation": "Notice how changing just one word at the end of the sentence completely alters the self-attention pattern for 'it'. Attention is dynamic and routes information based on semantic compatibility."
  },
  {
    "id": "subject-verb",
    "title": "Long-Distance Agreement: Keys to the Cabinet",
    "category": "Grammar & Syntax",
    "description": "The plural verb 'are' must attend over the singular distractor 'cabinet' to match plural 'keys'.",
    "sentence": "The keys to the cabinet are on the wooden table",
    "default_head": 0,
    "key_pairs": [
      {
        "source_token": "are",
        "source_idx": 5,
        "target_token": "keys",
        "target_idx": 1,
        "expected_strength": "high",
        "reason": "Plural auxiliary 'are' must agree in number with subject 'keys', ignoring singular noun 'cabinet'."
      }
    ],
    "explanation": "In the Syntax Head (Head 0), 'are' learns to skip the intervening prepositional phrase ('to the cabinet') and form a strong attention bridge directly back to 'keys'."
  },
  {
    "id": "induction-pattern",
    "title": "Anthropic Induction Circuit: Pattern Copying",
    "category": "Induction Circuit",
    "description": "How Transformers implement in-context learning: [A][B] ... [A] -> [B].",
    "sentence": "Harry Potter and Harry Potter",
    "default_head": 2,
    "key_pairs": [
      {
        "source_token": "Harry",
        "source_idx": 3,
        "target_token": "Potter",
        "target_idx": 1,
        "expected_strength": "high",
        "reason": "The induction head attends from the second 'Harry' back to 'Potter' (the token that followed the previous 'Harry') to predict continuation."
      }
    ],
    "explanation": "Discovered by Anthropic in 'In-context Learning and Induction Heads' (2022). When a token reappears, the induction head looks at earlier instances of that token and shifts attention to the subsequent token, enabling few-shot pattern copying."
  },
  {
    "id": "polysemy-disambiguation",
    "title": "Polysemy Disambiguation: River Bank vs Financial Bank",
    "category": "Contextual Disambiguation",
    "description": "Disambiguating homonyms through self-attention context exchange.",
    "sentence": "The river bank was muddy after the heavy rain",
    "default_head": 1,
    "key_pairs": [
      {
        "source_token": "bank",
        "source_idx": 2,
        "target_token": "river",
        "target_idx": 1,
        "expected_strength": "high",
        "reason": "'river' injects aquatic features into 'bank', removing financial ambiguity."
      },
      {
        "source_token": "bank",
        "source_idx": 2,
        "target_token": "muddy",
        "target_idx": 4,
        "expected_strength": "high",
        "reason": "'muddy' reinforces the geological soil interpretation."
      }
    ],
    "explanation": "Before self-attention, the embedding for 'bank' is generic. Once attention weights are computed, the value vectors of 'river' and 'muddy' are added into 'bank', enriching its hidden state representation so downstream layers know it is an embankment."
  },
  {
    "id": "code-variable-scope",
    "title": "Code & Logic: Variable Scoping in Functions",
    "category": "Code & Logic",
    "description": "Tracking identifier references inside programming code syntax.",
    "sentence": "def calculate_total ( items ) : return sum ( items )",
    "default_head": 0,
    "key_pairs": [
      {
        "source_token": "items",
        "source_idx": 9,
        "target_token": "items",
        "target_idx": 3,
        "expected_strength": "high",
        "reason": "Return expression identifier attends to function parameter declaration."
      }
    ],
    "explanation": "In code-specialized LLMs, attention heads specialize in linking variable uses back to their definitions across scope boundaries."
  }
];

// Heads Metadata
const HEADS_META = [
  {
    id: 0,
    name: "Head 0: Syntactic & Local Context",
    specialization: "Syntax",
    desc: "Attends to adjacent tokens, prepositions, and grammatical subject-verb linkages."
  },
  {
    id: 1,
    name: "Head 1: Coreference & Semantic Resolution",
    specialization: "Coreference",
    desc: "Resolves pronouns (it, they) to their semantic antecedents via context adjectives."
  },
  {
    id: 2,
    name: "Head 2: Induction Circuit (Pattern Copying)",
    specialization: "Induction / Pattern Copying",
    desc: "Implements in-context learning: looks back at prior occurrences to copy succeeding tokens."
  },
  {
    id: 3,
    name: "Head 3: Global Anchor & Prefix Head",
    specialization: "Global Anchor",
    desc: "Anchors information to sentence root or prompt origin tokens."
  }
];

// App State
const state = {
  presets: FALLBACK_PRESETS,
  activePresetId: "winograd-tired",
  sentence: "The animal didn't cross the street because it was too tired",
  tokens: [],
  activeHead: 1, // 0, 1, 2, 3 or -1 for composite
  isCausal: false,
  temperature: 1.0,
  useScaling: true,
  d_k: 8,
  attentionResult: null,
  isolatedTokenIdx: null
};

// ============================================================================
// Core Attention Mathematics (Client-Side)
// ============================================================================

function tokenize(text) {
  const cleaned = text.trim();
  if (!cleaned) return [];
  const pattern = /\b[\w']+\b|[^\w\s]/g;
  const matches = cleaned.match(pattern);
  return matches || cleaned.split(/\s+/);
}

function stableSoftmax(scores, temperature = 1.0) {
  const t = Math.max(0.01, temperature);
  const validScores = scores.filter(s => s > -1e8);
  if (validScores.length === 0) {
    return new Array(scores.length).fill(0.0);
  }

  const maxVal = Math.max(...validScores);
  const exps = scores.map(s => (s <= -1e8 ? 0.0 : Math.exp((s - maxVal) / t)));
  const sum = exps.reduce((acc, v) => acc + v, 0);

  if (sum <= 0) {
    return new Array(scores.length).fill(1.0 / scores.length);
  }
  return exps.map(v => v / sum);
}

function generateHeadVectors(tokens, headIdx, d_k) {
  const n = tokens.length;
  const Q = [];
  const K = [];
  const V = [];

  const tokensLower = tokens.map(t => t.toLowerCase());
  const hasTired = tokensLower.includes("tired");
  const hasWide = tokensLower.includes("wide");

  for (let i = 0; i < n; i++) {
    const tok = tokensLower[i];
    let seed = 0;
    for (let p = 0; p < tok.length; p++) {
      seed += tok.charCodeAt(p) * (p + 1);
    }

    const qVec = [];
    const kVec = [];
    const vVec = [];

    for (let d = 0; d < d_k; d++) {
      qVec.push(Math.sin(seed + d * 1.31));
      kVec.push(Math.cos(seed + d * 1.31));
      vVec.push(Math.sin(seed * 0.5 + d * 0.77));
    }

    // Role-specific specialization
    if (headIdx === 0) {
      // Syntax & local
      qVec[0] = i * 0.5;
      kVec[0] = i * 0.5;
      if (["are", "is", "was", "were"].includes(tok)) qVec[1] = 3.0;
      if (["keys", "animal", "street", "table", "cabinet"].includes(tok)) {
        kVec[1] = tok === "keys" ? 2.5 : 0.5;
      }
    } else if (headIdx === 1) {
      // Coreference
      if (["it", "they", "he", "she", "this", "that"].includes(tok)) {
        qVec[0] = 4.0;
        qVec[1] = hasTired ? 4.0 : (hasWide ? -4.0 : 1.0);
      }
      if (tok === "animal") {
        kVec[0] = 3.5;
        kVec[1] = 4.0;
      } else if (tok === "street") {
        kVec[0] = 3.5;
        kVec[1] = -4.0;
      }

      if (tok === "bank") qVec[2] = 3.0;
      if (["river", "muddy", "water", "rain"].includes(tok)) kVec[2] = 3.5;
    } else if (headIdx === 2) {
      // Induction circuit: match token after prior occurrence
      const prevOccurrences = [];
      for (let prev = 0; prev < i; prev++) {
        if (tokensLower[prev] === tok) prevOccurrences.push(prev);
      }
      if (prevOccurrences.length > 0) {
        qVec[3] = 5.0;
        const targetIdx = prevOccurrences[prevOccurrences.length - 1] + 1;
        if (targetIdx < n) {
          const targetTok = tokensLower[targetIdx];
          let tSeed = 0;
          for (let p = 0; p < targetTok.length; p++) {
            tSeed += targetTok.charCodeAt(p) * (p + 1);
          }
          for (let d = 0; d < d_k; d++) {
            qVec[d] = Math.cos(tSeed + d * 1.31) * 3.0;
          }
        }
      } else {
        qVec[3] = -1.0;
      }
    } else if (headIdx === 3) {
      // Global root anchor
      qVec[4 % d_k] = 3.0;
      kVec[4 % d_k] = (i === 0) ? 5.0 : 0.5;
    }

    // Normalize
    const qNorm = Math.sqrt(qVec.reduce((sum, x) => sum + x * x, 0)) || 1.0;
    const kNorm = Math.sqrt(kVec.reduce((sum, x) => sum + x * x, 0)) || 1.0;
    const vNorm = Math.sqrt(vVec.reduce((sum, x) => sum + x * x, 0)) || 1.0;

    Q.push(qVec.map(x => (x / qNorm) * 2.0));
    K.push(kVec.map(x => (x / kNorm) * 2.0));
    V.push(vVec.map(x => (x / vNorm) * 1.5));
  }

  return { Q, K, V };
}

function computeSelfAttention(sentence, isCausal, temperature, useScaling) {
  const tokens = tokenize(sentence);
  const n = tokens.length;
  const d_k = state.d_k;
  const scale = useScaling ? Math.sqrt(d_k) : 1.0;

  if (n === 0) {
    return { tokens: [], heads: [], composite: [] };
  }

  const heads = [];
  const composite = Array.from({ length: n }, () => new Array(n).fill(0.0));

  for (let h = 0; h < 4; h++) {
    const { Q, K, V } = generateHeadVectors(tokens, h, d_k);
    const rawScores = [];
    const scaledScores = [];
    const weights = [];
    const outputs = [];

    for (let i = 0; i < n; i++) {
      const rowRaw = [];
      const rowScaled = [];

      for (let j = 0; j < n; j++) {
        if (isCausal && j > i) {
          rowRaw.push(-1e9);
          rowScaled.push(-1e9);
        } else {
          let dot = 0.0;
          for (let d = 0; d < d_k; d++) {
            dot += Q[i][d] * K[j][d];
          }
          rowRaw.push(dot);
          rowScaled.push(dot / scale);
        }
      }

      const wRow = stableSoftmax(rowScaled, temperature);
      rawScores.push(rowRaw);
      scaledScores.push(rowScaled);
      weights.push(wRow);

      // Value output
      const outVec = new Array(d_k).fill(0.0);
      for (let j = 0; j < n; j++) {
        const w = wRow[j];
        for (let d = 0; d < d_k; d++) {
          outVec[d] += w * V[j][d];
        }
      }
      outputs.push(outVec);

      for (let j = 0; j < n; j++) {
        composite[i][j] += wRow[j] / 4.0;
      }
    }

    heads.push({
      meta: HEADS_META[h],
      Q,
      K,
      V,
      rawScores,
      scaledScores,
      weights,
      outputs
    });
  }

  return { tokens, heads, composite };
}

// ============================================================================
// UI Renderers
// ============================================================================

function getActiveMatrix() {
  if (!state.attentionResult) return [];
  if (state.activeHead === -1) {
    return state.attentionResult.composite;
  }
  return state.attentionResult.heads[state.activeHead].weights;
}

function renderTokensChips() {
  const container = document.getElementById("tokens-chip-list");
  const countBadge = document.getElementById("token-count-badge");
  container.innerHTML = "";

  const tokens = state.attentionResult.tokens;
  countBadge.textContent = `${tokens.length} Tokens`;

  tokens.forEach((tok, idx) => {
    const chip = document.createElement("div");
    chip.className = `token-chip ${state.isolatedTokenIdx === idx ? "active" : ""}`;
    chip.innerHTML = `<span class="tok-idx">${idx}</span><span>${tok}</span>`;
    chip.addEventListener("click", () => {
      state.isolatedTokenIdx = (state.isolatedTokenIdx === idx) ? null : idx;
      renderAll();
    });
    container.appendChild(chip);
  });
}

function renderHeatmapMatrix() {
  const grid = document.getElementById("matrix-grid");
  const matrixTitle = document.getElementById("matrix-title");
  grid.innerHTML = "";

  if (state.activeHead === -1) {
    matrixTitle.textContent = "Composite Attention Matrix (Average across all 4 Heads)";
  } else {
    const meta = HEADS_META[state.activeHead];
    matrixTitle.textContent = `${meta.name} (${meta.specialization})`;
  }

  const tokens = state.attentionResult.tokens;
  const n = tokens.length;
  const matrix = getActiveMatrix();

  // Set grid template columns: 1 header col + n data cols
  grid.style.gridTemplateColumns = `80px repeat(${n}, 48px)`;

  // Top-left corner cell
  const corner = document.createElement("div");
  corner.className = "grid-cell grid-header-cell";
  corner.innerHTML = "Q \\ K";
  grid.appendChild(corner);

  // Column header cells (Keys)
  tokens.forEach((tok, j) => {
    const colHeader = document.createElement("div");
    colHeader.className = "grid-cell grid-header-cell col-header";
    colHeader.title = `Key [${j}]: ${tok}`;
    colHeader.textContent = tok.length > 5 ? tok.substring(0, 4) + "…" : tok;
    grid.appendChild(colHeader);
  });

  // Rows (Queries)
  for (let i = 0; i < n; i++) {
    // Row Header (Query)
    const rowHeader = document.createElement("div");
    rowHeader.className = "grid-cell grid-header-cell row-header";
    rowHeader.title = `Query [${i}]: ${tokens[i]}`;
    rowHeader.textContent = tokens[i].length > 6 ? tokens[i].substring(0, 5) + "…" : tokens[i];
    grid.appendChild(rowHeader);

    // Matrix cells
    for (let j = 0; j < n; j++) {
      const cell = document.createElement("div");
      const val = matrix[i][j];
      const isMasked = state.isCausal && j > i;

      cell.className = `grid-cell ${isMasked ? "masked" : ""}`;
      cell.dataset.row = i;
      cell.dataset.col = j;

      if (isMasked) {
        cell.textContent = "—";
        cell.title = `Masked [${i} ➔ ${j}]: -∞ (Causal Future Token)`;
      } else {
        cell.textContent = (val < 0.01) ? "·" : (val >= 0.99 ? "1.0" : val.toFixed(2).replace(/^0/, ""));
        // Color mapping
        const bg = interpolateAttentionColor(val);
        cell.style.backgroundColor = bg;
        cell.style.color = (val > 0.5) ? "#ffffff" : "rgba(255, 255, 255, 0.75)";
        cell.title = `Query '${tokens[i]}' [${i}] ➔ Key '${tokens[j]}' [${j}]: ${(val * 100).toFixed(1)}%`;
      }

      cell.addEventListener("mouseenter", () => updateMathInspector(i, j));
      grid.appendChild(cell);
    }
  }
}

function interpolateAttentionColor(weight) {
  // 0.0 -> #0f172a (dark)
  // 0.25 -> #0369a1 (blue)
  // 0.6 -> #06b6d4 (cyan)
  // 0.85 -> #6366f1 (indigo)
  // 1.0 -> #ec4899 (magenta/pink)
  const w = Math.max(0, Math.min(1, weight));
  if (w < 0.05) return "rgba(15, 23, 42, 0.9)";
  if (w < 0.3) {
    const t = w / 0.3;
    return `rgba(3, 105, 161, ${0.3 + t * 0.4})`;
  }
  if (w < 0.7) {
    const t = (w - 0.3) / 0.4;
    return `rgba(6, 182, 212, ${0.4 + t * 0.45})`;
  }
  const t = (w - 0.7) / 0.3;
  return `rgba(236, 72, 153, ${0.7 + t * 0.3})`;
}

function updateMathInspector(qIdx, kIdx) {
  const tokens = state.attentionResult.tokens;
  const qTok = tokens[qIdx];
  const kTok = tokens[kIdx];
  const isMasked = state.isCausal && kIdx > qIdx;

  document.getElementById("insp-query-tok").textContent = `${qTok} [${qIdx}]`;
  document.getElementById("insp-key-tok").textContent = `${kTok} [${kIdx}]`;

  const headIdx = (state.activeHead === -1) ? 0 : state.activeHead;
  const headData = state.attentionResult.heads[headIdx];

  const rawScore = headData.rawScores[qIdx][kIdx];
  const scaledScore = headData.scaledScores[qIdx][kIdx];
  const finalWeight = getActiveMatrix()[qIdx][kIdx];

  if (isMasked) {
    document.getElementById("insp-raw-score").textContent = "-∞ (Masked)";
    document.getElementById("insp-scaled-score").textContent = "-∞";
    document.getElementById("insp-temp-val").textContent = "-∞";
    document.getElementById("insp-final-weight").textContent = "0.000 (0.0%)";
    document.getElementById("insp-note").textContent = 
      "🔒 Causal Mask active: Decoder models (GPT) strictly mask future tokens to prevent information leaking backward in time.";
    return;
  }

  document.getElementById("insp-raw-score").textContent = rawScore.toFixed(3);
  document.getElementById("insp-scaled-score").textContent = scaledScore.toFixed(3);
  document.getElementById("insp-temp-val").textContent = (scaledScore / state.temperature).toFixed(3);
  document.getElementById("insp-final-weight").textContent = 
    `${finalWeight.toFixed(3)} (${(finalWeight * 100).toFixed(1)}%)`;

  // Contextual Note
  let note = `Attention from '${qTok}' to '${kTok}'.`;
  if (finalWeight > 0.45) {
    note = `🌟 Dominant attention bond! Query '${qTok}' directs ${(finalWeight * 100).toFixed(1)}% of its attention context directly to '${kTok}'.`;
  } else if (finalWeight > 0.20) {
    note = `✨ Moderate attention link. '${qTok}' extracts partial contextual information from '${kTok}'.`;
  } else {
    note = `Minimal attention weight (${(finalWeight * 100).toFixed(1)}%). Most attention was captured by more semantically relevant tokens.`;
  }
  document.getElementById("insp-note").textContent = note;
}

// ============================================================================
// Bipartite Flow Graph (SVG Rays & Arcs)
// ============================================================================

function renderBipartiteFlow() {
  const queryCol = document.getElementById("query-chips");
  const keyCol = document.getElementById("key-chips");
  const svg = document.getElementById("attention-flow-svg");

  queryCol.innerHTML = "";
  keyCol.innerHTML = "";
  svg.innerHTML = "";

  const tokens = state.attentionResult.tokens;
  const n = tokens.length;
  const matrix = getActiveMatrix();

  // Create Query & Key chips
  tokens.forEach((tok, idx) => {
    const qChip = document.createElement("div");
    qChip.className = `flow-chip ${state.isolatedTokenIdx === idx ? "active" : ""}`;
    qChip.id = `q-chip-${idx}`;
    qChip.innerHTML = `<span>${tok}</span><span style="opacity:0.6;font-size:0.7rem">#${idx}</span>`;
    qChip.addEventListener("mouseenter", () => highlightFlowToken(idx, "query"));
    qChip.addEventListener("mouseleave", () => unhighlightFlowToken());
    queryCol.appendChild(qChip);

    const kChip = document.createElement("div");
    kChip.className = "flow-chip";
    kChip.id = `k-chip-${idx}`;
    kChip.innerHTML = `<span style="opacity:0.6;font-size:0.7rem">#${idx}</span><span>${tok}</span>`;
    kChip.addEventListener("mouseenter", () => highlightFlowToken(idx, "key"));
    kChip.addEventListener("mouseleave", () => unhighlightFlowToken());
    keyCol.appendChild(kChip);
  });

  // Wait for DOM positions to calculate SVG coordinates
  requestAnimationFrame(() => {
    drawFlowRays(tokens, matrix);
  });
}

function drawFlowRays(tokens, matrix) {
  const svg = document.getElementById("attention-flow-svg");
  const wrapper = svg.parentElement;
  const rect = wrapper.getBoundingClientRect();
  const n = tokens.length;

  svg.setAttribute("viewBox", `0 0 ${rect.width} ${rect.height}`);
  svg.innerHTML = "";

  for (let i = 0; i < n; i++) {
    const qEl = document.getElementById(`q-chip-${i}`);
    if (!qEl) continue;
    const qRect = qEl.getBoundingClientRect();
    const y1 = qRect.top - rect.top + qRect.height / 2;
    const x1 = 0;

    for (let j = 0; j < n; j++) {
      const kEl = document.getElementById(`k-chip-${j}`);
      if (!kEl) continue;
      const kRect = kEl.getBoundingClientRect();
      const y2 = kRect.top - rect.top + kRect.height / 2;
      const x2 = rect.width;

      const weight = matrix[i][j];
      const isMasked = state.isCausal && j > i;
      if (isMasked || weight < 0.015) continue;

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const dx = (x2 - x1) * 0.5;
      const d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;

      path.setAttribute("d", d);
      path.setAttribute("class", "attention-ray");
      path.dataset.query = i;
      path.dataset.key = j;
      path.dataset.weight = weight;

      // Styling based on weight
      const strokeW = Math.max(0.6, weight * 5.0);
      const opacity = Math.max(0.08, Math.min(0.9, weight * 1.2));
      path.style.strokeWidth = `${strokeW}px`;
      path.style.opacity = `${opacity}`;

      if (weight > 0.40) {
        path.style.stroke = "#ec4899"; // High attention hot pink
      } else if (weight > 0.20) {
        path.style.stroke = "#06b6d4"; // Moderate cyan
      } else {
        path.style.stroke = "#6366f1"; // Standard indigo
      }

      path.addEventListener("mouseenter", () => updateMathInspector(i, j));
      svg.appendChild(path);
    }
  }
}

function highlightFlowToken(targetIdx, type) {
  const rays = document.querySelectorAll(".attention-ray");
  rays.forEach(r => {
    const qIdx = parseInt(r.dataset.query, 10);
    const kIdx = parseInt(r.dataset.key, 10);

    const isMatch = (type === "query" && qIdx === targetIdx) || (type === "key" && kIdx === targetIdx);
    if (isMatch) {
      r.style.opacity = "1.0";
      r.style.strokeWidth = "3.5px";
      r.style.stroke = "#ec4899";
    } else {
      r.style.opacity = "0.04";
      r.style.strokeWidth = "0.5px";
    }
  });
}

function unhighlightFlowToken() {
  const rays = document.querySelectorAll(".attention-ray");
  rays.forEach(r => {
    const weight = parseFloat(r.dataset.weight);
    r.style.strokeWidth = `${Math.max(0.6, weight * 5.0)}px`;
    r.style.opacity = `${Math.max(0.08, Math.min(0.9, weight * 1.2))}`;
    if (weight > 0.40) {
      r.style.stroke = "#ec4899";
    } else if (weight > 0.20) {
      r.style.stroke = "#06b6d4";
    } else {
      r.style.stroke = "#6366f1";
    }
  });
}

// ============================================================================
// Presets & Insights Panel
// ============================================================================

function renderPresetInsights(preset) {
  if (!preset) return;
  document.getElementById("scenario-title").textContent = preset.title;
  document.getElementById("scenario-explanation").textContent = preset.explanation;

  const keyPairsGrid = document.getElementById("key-pairs-display");
  keyPairsGrid.innerHTML = "";

  preset.key_pairs.forEach(pair => {
    const card = document.createElement("div");
    card.className = "key-pair-card";
    card.innerHTML = `
      <div class="pair-badge-line">
        <span class="pair-source">${pair.source_token} [#${pair.source_idx}]</span>
        <span>➔</span>
        <span class="pair-target">${pair.target_token} [#${pair.target_idx}]</span>
        <span class="badge-status" style="font-size:0.65rem">${pair.expected_strength.toUpperCase()}</span>
      </div>
      <div class="pair-reason">${pair.reason}</div>
    `;
    keyPairsGrid.appendChild(card);
  });
}

function selectPreset(presetId) {
  const preset = state.presets.find(p => p.id === presetId);
  if (!preset) return;

  state.activePresetId = presetId;
  state.sentence = preset.sentence;
  state.activeHead = preset.default_head;
  state.isolatedTokenIdx = null;

  document.getElementById("sentence-input").value = preset.sentence;

  // Update preset buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.id === presetId);
  });

  // Update head pills
  updateHeadPillsUI();

  // Recompute & render
  computeAndRender();
  renderPresetInsights(preset);
}

function updateHeadPillsUI() {
  document.querySelectorAll(".head-pill").forEach(pill => {
    const headId = parseInt(pill.dataset.head, 10);
    pill.classList.toggle("active", headId === state.activeHead);
  });

  const descEl = document.getElementById("head-desc-display");
  if (state.activeHead === -1) {
    descEl.textContent = "Composite View: Arithmetic average of attention weights across all 4 specialized heads.";
  } else {
    const meta = HEADS_META[state.activeHead];
    descEl.textContent = `${meta.name}: ${meta.desc}`;
  }
}

// ============================================================================
// Master Render Cycle & Event Listeners
// ============================================================================

function computeAndRender() {
  state.attentionResult = computeSelfAttention(
    state.sentence,
    state.isCausal,
    state.temperature,
    state.useScaling
  );
  renderAll();
}

function renderAll() {
  renderTokensChips();
  renderHeatmapMatrix();
  renderBipartiteFlow();

  // Initialize inspector with first query or notable pair
  if (state.attentionResult.tokens.length > 0) {
    const tokens = state.attentionResult.tokens;
    let defQ = 0;
    let defK = 0;
    // Prefer pronoun 'it' if exists
    const itIdx = tokens.findIndex(t => t.toLowerCase() === "it");
    if (itIdx !== -1) {
      defQ = itIdx;
      defK = tokens.findIndex(t => ["animal", "street"].includes(t.toLowerCase()));
      if (defK === -1) defK = 0;
    }
    updateMathInspector(defQ, defK);
  }
}

function setupEventListeners() {
  // Compute Button & Enter key
  const computeBtn = document.getElementById("compute-btn");
  const sentenceInput = document.getElementById("sentence-input");

  computeBtn.addEventListener("click", () => {
    const val = sentenceInput.value.trim();
    if (val) {
      state.sentence = val;
      state.activePresetId = null;
      document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
      computeAndRender();
    }
  });

  sentenceInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      computeBtn.click();
    }
  });

  // Preset Buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      selectPreset(btn.dataset.id);
    });
  });

  // Head Selector Pills
  document.querySelectorAll(".head-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      state.activeHead = parseInt(pill.dataset.head, 10);
      updateHeadPillsUI();
      renderAll();
    });
  });

  // Causal Mode Toggles
  const btnBidir = document.getElementById("btn-bidirectional");
  const btnCausal = document.getElementById("btn-causal");

  btnBidir.addEventListener("click", () => {
    state.isCausal = false;
    btnBidir.classList.add("active");
    btnCausal.classList.remove("active");
    computeAndRender();
  });

  btnCausal.addEventListener("click", () => {
    state.isCausal = true;
    btnCausal.classList.add("active");
    btnBidir.classList.remove("active");
    computeAndRender();
  });

  // Scaling Factor Toggle
  const scalingToggle = document.getElementById("scaling-toggle");
  scalingToggle.addEventListener("change", (e) => {
    state.useScaling = e.target.checked;
    computeAndRender();
  });

  // Temperature Slider
  const tempSlider = document.getElementById("temp-slider");
  const tempVal = document.getElementById("temp-val");
  tempSlider.addEventListener("input", (e) => {
    state.temperature = parseFloat(e.target.value);
    tempVal.textContent = state.temperature.toFixed(2);
    computeAndRender();
  });

  // Reset Flow Button
  document.getElementById("reset-flow-btn").addEventListener("click", () => {
    state.isolatedTokenIdx = null;
    unhighlightFlowToken();
    renderTokensChips();
  });

  // Window Resize re-draws SVG rays
  window.addEventListener("resize", () => {
    if (state.attentionResult) {
      drawFlowRays(state.attentionResult.tokens, getActiveMatrix());
    }
  });
}

// Load presets.json or fallback
async function initApp() {
  try {
    const res = await fetch("data/presets.json");
    if (res.ok) {
      const presets = await res.json();
      if (Array.isArray(presets) && presets.length > 0) {
        state.presets = presets;
      }
    }
  } catch {
    console.info("Using embedded fallback presets (likely served via file:// protocol).");
  }

  setupEventListeners();
  selectPreset("winograd-tired");
}

// Initialize when DOM ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}
