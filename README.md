# LLM Multi-Head Self-Attention Lab

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Pages Ready](https://img.shields.io/badge/GitHub_Pages-Ready-brightgreen.svg)](#-how-to-host-on-github-pages)
[![Tests Passing](https://img.shields.io/badge/Tests-16%20Passed-emerald.svg)](#-local-testing--verification)

An interactive, zero-backend, client-side web application designed to be hosted directly on **GitHub Pages**. 
It visually and mathematically unpacks how **Multi-Head Self-Attention** operates inside Large Language Models, directly inspired by:
- 🎨 **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** (Jay Alammar)
- 🎓 **[Transformer Explainer](https://poloclub.github.io/transformer-explainer/)** (Georgia Tech Polo Club)
- 🔬 **[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)** (Anthropic Transformer Circuits)

---

## 🌐 Live GitHub Pages Demo
- **Live Interactive Demo:** [https://martianunlimited.github.io/llm_attention/](https://martianunlimited.github.io/llm_attention/)
- **Companion Tokenizer Demo:** [https://martianunlimited.github.io/llm_tokenizer/](https://martianunlimited.github.io/llm_tokenizer/)
- **Companion Embeddings Demo:** [https://martianunlimited.github.io/llm_embedding/](https://martianunlimited.github.io/llm_embedding/)
- **GitHub Repository:** [https://github.com/martianunlimited/llm_attention](https://github.com/martianunlimited/llm_attention)

---

## 🧭 Interactive Modules in the Demo

### 1. Scenario & Sentence Studio
* **Winograd Schema Ambiguity:**
  - *"The animal didn't cross the street because it was too tired"* $\to$ **it** attends strongly to **animal**.
  - *"The animal didn't cross the street because it was too wide"* $\to$ **it** shifts attention directly to **street**.
* **Long-Distance Syntactic Agreement:**
  - *"The keys to the cabinet are on the table"* $\to$ Plural verb **are** skips the intervening singular noun **cabinet** to link directly with **keys**.
* **Anthropic Induction Circuit (In-Context Learning):**
  - *"Harry Potter and Harry Potter"* $\to$ The induction head detects the repeating token and attends to the subsequent token to predict pattern continuations ($[A][B] \dots [A] \to [B]$).
* **Polysemy Disambiguation:**
  - *"The river bank was muddy"* $\to$ Aquatic and soil words enrich the hidden representation of **bank**, suppressing the financial interpretation.

### 2. The $N \times N$ Attention Matrix Heatmap
* Live scaled dot-product and softmax probability grid.
* Cell hover reveals exact arithmetic:
  $$\text{Score}(i, j) = \frac{\vec{q}_i \cdot \vec{k}_j}{\sqrt{d_k}} \quad \xrightarrow{\text{softmax}} \quad \alpha_{ij} = \frac{e^{\text{Score}(i, j) / T}}{\sum_m e^{\text{Score}(i, m) / T}}$$
* **Causal Masking Toggle:** Upper-triangular positions set to $-\infty$ (rendered with dark hatched stripes) to show how GPT autoregressive models prevent future token leakage.

### 3. Bipartite Attention Flow Graph (Rays & Arcs)
* SVG cubic bezier curves connecting Query tokens on the left with Key tokens on the right.
* Stroke width and opacity scale dynamically with attention weight.
* Hover over any word to isolate its incoming or outgoing attention rays while dimming all distractors.

### 4. Step-by-Step Transformer Pipeline
* Visualizes the 6 stages of the attention forward pass:
  1. Input Embeddings $X$
  2. Linear Projections ($W_Q, W_K, W_V$)
  3. Raw Dot Product $QK^T$
  4. Scale by $\sqrt{d_k}$
  5. Softmax Normalization
  6. Value Weighted Accumulation $\vec{o}_i = \sum \alpha_{ij} \vec{v}_j$ and Multi-Head Concatenation ($W_O$)

---

## 🚀 How to Host on GitHub Pages

This project is 100% static (HTML, CSS, JS) and requires **no build step, no Node server, and no Python backend**:

```bash
# 1. Initialize git and commit
git init
git add .
git commit -m "Initial commit: LLM Multi-Head Self-Attention Lab"

# 2. Push to your GitHub repository
git branch -M main
git remote add origin https://github.com/martianunlimited/llm_attention.git
git push -u origin main

# 3. Enable GitHub Pages in your repo:
# Settings -> Pages -> Source: Deploy from a branch -> Branch: main -> / (root) -> Save!
# Your live site is now ready at: https://martianunlimited.github.io/llm_attention/
```

---

## 🧪 Local Testing & Verification

### 1. Run Automated Tests
```bash
python -m pytest -q
# 16 passed in 0.19s

python -m ruff check . --fix
python -m ruff format .
```

### 2. Terminal CLI Explorer
```bash
# Inspect available educational presets:
python -m src.cli presets

# Compute self-attention for a sentence:
python -m src.cli compute "The animal didn't cross the street because it was too tired" --head 1

# Inspect detailed scenario explanation:
python -m src.cli inspect winograd-tired
```

### 3. Preview Locally
```bash
python -m http.server 8080
# Open http://localhost:8080 in your browser
```

---

## 📄 License
MIT License. Free for educational, academic, and open-source use.
