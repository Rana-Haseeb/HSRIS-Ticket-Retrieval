# 🔎 HSRIS — Hybrid Semantic Retrieval & Intelligence System

**A customer-support ticket retrieval engine built entirely from first principles — no `scikit-learn`, no pre-built vectorizers, no shortcuts.**

Every encoder, tokenizer, TF-IDF vectorizer, and similarity engine here is hand-implemented in pure **NumPy + PyTorch**, then fused with **GloVe 300D semantic embeddings** into a dual-channel hybrid search system, GPU-accelerated across dual Tesla T4s, and shipped with a live **Gradio** dashboard.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Sparse%20COO-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Gradio" src="https://img.shields.io/badge/Gradio-UI-FF7C00?logo=gradio&logoColor=white">
  <img alt="No sklearn" src="https://img.shields.io/badge/scikit--learn-not%20used-critical">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-Dual%20T4%20DataParallel-76B900?logo=nvidia&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-Academic%2FEducational-lightgrey">
</p>

---

## 🧠 What Is This?

HSRIS is an assignment-turned-engine that tackles a real information-retrieval problem: **given a new customer support ticket, find the most similar historical tickets** — using not one, but *two* complementary retrieval strategies blended together:

| Channel | Technique | Captures |
|---|---|---|
| 📝 **Sparse (Lexical)** | Custom TF-IDF over 1/2/3-grams → PyTorch sparse COO tensor | Exact keyword & phrase overlap |
| 🧬 **Dense (Semantic)** | GloVe 300D + IDF-weighted averaging | Meaning & intent, even with zero shared words |
| 🔀 **Hybrid** | `score = α · dense + (1-α) · sparse` | Best of both worlds, tunable live |

The result is a system that can match *"my laptop screen is flickering"* to a ticket about *"display flashing repeatedly"* — a case where keyword search fails but semantic search wins.

---

## 🏗️ Architecture

```
                       ┌─────────────────────────────┐
                       │   Raw Support Ticket Data    │
                       │  (subject, description, ...) │
                       └──────────────┬───────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────────┐         ┌───────────────────┐
│  Categorical   │           │   Sparse Channel   │         │   Dense Channel    │
│   Encoders     │           │  (Custom TF-IDF)   │         │  (GloVe 300D +     │
│                │           │                    │         │   IDF-weighted     │
│ • LabelEncoder │           │ • Tokenizer        │         │   averaging)       │
│ • OneHotEncoder│           │ • 1/2/3-gram gen   │         │                    │
│ (with <UNK>    │           │ • Smooth IDF        │        │ • Frozen nn.Embed  │
│  safety net)   │           │ • Sparse COO tensor │        │ • <UNK> fallback   │
└───────────────┘           └─────────┬──────────┘         └─────────┬──────────┘
                                       │                              │
                                       ▼                              ▼
                            ┌─────────────────────────────────────────────┐
                            │        SimilarityScorer (nn.Module)         │
                            │   Cosine sim via matmul on L2-normed vecs   │
                            │   Registered buffers → DataParallel-ready   │
                            │        Dual T4 GPU (torch.nn.DataParallel)  │
                            └─────────────────────┬─────────────────────┘
                                                   │
                                                   ▼
                                   score = α·dense + (1-α)·sparse
                                                   │
                                                   ▼
                                        Top-K Ranked Results
                                                   │
                                                   ▼
                                   ┌───────────────────────────┐
                                   │   Gradio Web Dashboard     │
                                   │  (live α slider, 3-way     │
                                   │   channel comparison)      │
                                   └───────────────────────────┘
```

---

## ✨ Key Features

- **🚫 Zero `scikit-learn` dependency** — every ML primitive (label encoding, one-hot encoding, TF-IDF, cosine similarity) is implemented from scratch.
- **🛡️ `<UNK>` safety nets everywhere** — unseen categories, unseen tokens, and out-of-vocabulary words never crash inference; they gracefully fall back to a dedicated unknown token/vector.
- **💾 Memory-efficient sparse storage** — TF-IDF is stored as a PyTorch sparse COO tensor, cutting an ~170MB dense matrix down to a few MB by keeping only non-zero entries.
- **🧬 Semantic understanding via GloVe** — 300-dimensional pretrained word vectors, combined with IDF weights so important words dominate the sentence embedding.
- **🔀 Tunable hybrid scoring** — a single `alpha` parameter blends lexical and semantic relevance, from pure keyword match (`α=0`) to pure semantic match (`α=1`).
- **⚡ Dual-GPU acceleration** — the similarity scorer is wrapped in `torch.nn.DataParallel` to exploit dual Tesla T4 GPUs (Kaggle-friendly).
- **📊 Built-in benchmarking** — throughput (queries/sec) measured across multiple batch sizes, plus GPU memory reporting.
- **📈 Quantitative evaluation** — Precision@5 computed over 100 sampled queries, measuring how often retrieved tickets share the query's true `Ticket Type`.
- **🔍 Qualitative case studies** — automatically mines examples where semantic (GloVe) retrieval succeeds while pure keyword (TF-IDF) retrieval fails, proving the value of the semantic channel.
- **🖥️ Interactive Gradio dashboard** — search live, adjust α with a slider, and compare Blended / TF-IDF-only / GloVe-only results side by side.

---

## 📂 Project Structure

```
HSRIS-Ticket-Retrieval/
├── DS_ASS03_23F_3096.ipynb   # The entire pipeline, in 6 self-contained parts
└── README.md                  # You are here
```

### Notebook Breakdown

| Part | Section | What Happens |
|:---:|---|---|
| **1** | Categorical Foundation | `CustomLabelEncoder` & `CustomOneHotEncoder` built from scratch, encoding `Ticket Priority` and `Ticket Channel` with `<UNK>` handling |
| **2** | Sparse Representation | Custom tokenizer → n-gram generator (1/2/3-grams) → `CustomTfidfVectorizer` producing an L2-normalized PyTorch sparse COO tensor |
| **3** | Dense Semantic Layer | GloVe 6B/300D vectors loaded into a frozen `nn.Embedding`, combined with TF-IDF weights to build IDF-weighted sentence embeddings |
| **4** | Hybrid Search & GPU Optimization | `SimilarityScorer` module computes blended cosine similarity on GPU (DataParallel across dual T4s); includes a full retrieval function and batch throughput benchmark |
| **5** | Visualization & Deployment | A polished `Gradio` UI exposing live hybrid search with an α slider and 3-way channel comparison |
| **6** | Evaluation | Precision@5 scoring over 100 sampled tickets + automatically discovered examples where semantic search beats keyword search |

---

## 📦 Dataset

Trained and evaluated on the **[Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)** (Kaggle), containing ticket subjects, descriptions, priorities, channels, and types.

Semantic embeddings are powered by **[GloVe 6B 300D](https://www.kaggle.com/datasets/thanakomsn/glove6b300dtxt)** pretrained word vectors.

> The notebook auto-discovers both datasets via `glob`, so paths only need to point at the parent directory.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas torch gradio
```

> GPU acceleration (dual T4 / `DataParallel`) is optional — the code automatically falls back to CPU if no CUDA devices are found.

### Running It

1. Download the [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) and [GloVe 6B 300D](https://www.kaggle.com/datasets/thanakomsn/glove6b300dtxt) embeddings (this project was built for the Kaggle Notebooks environment, where both are attached as input datasets).
2. Update `DATA_DIR` and `GLOVE_DIR` in the notebook if running outside Kaggle.
3. Run all cells top to bottom — each part builds on objects created in the previous one (`df`, `vectorizer`, `glove_emb`, `scorer`, `hybrid_retrieve`).
4. The final cell of Part 5 launches an interactive Gradio dashboard (`demo.launch(share=True)`) for live querying.

### Example Usage

```python
results = hybrid_retrieve(
    "My laptop screen is flickering and the battery drains fast",
    alpha=0.5,   # 0 = pure keyword, 1 = pure semantic
    top_k=5,
)

for r in results:
    print(f"#{r['rank']}  score={r['score']:.4f}  {r['subject']}")
```

---

## 📊 Evaluation Snapshot

The notebook reports, over a 100-query random sample:

- **Precision@5** — fraction of top-5 retrieved tickets sharing the query's true `Ticket Type`, with mean ± standard deviation.
- **Throughput benchmarks** — queries/sec across batch sizes of 10 / 50 / 100 on the dual-GPU scorer.
- **Qualitative wins** — concrete before/after examples where GloVe's semantic understanding rescues a query that pure TF-IDF keyword matching gets wrong.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Numerical computing | NumPy |
| Data handling | Pandas |
| Tensors, embeddings, GPU compute | PyTorch (`nn.Embedding`, sparse COO, `DataParallel`) |
| Word vectors | GloVe 6B 300D |
| UI | Gradio (`Blocks` API, custom CSS theming) |
| Hardware | Dual NVIDIA Tesla T4 (Kaggle) |

---

## 🎓 About

Built as **Assignment 3** for a Data Science course, with an emphasis on implementing classic IR/NLP techniques (encoding, TF-IDF, embeddings, cosine similarity) manually rather than relying on high-level libraries — demonstrating a first-principles understanding of how modern hybrid search systems actually work under the hood.

---

<p align="center"><em>Built from first principles — No scikit-learn • Custom TF-IDF (PyTorch Sparse COO) • GloVe 300D + IDF-Weighted Embeddings • Dual T4 GPU DataParallel</em></p>
