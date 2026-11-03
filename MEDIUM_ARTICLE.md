# Beyond Cosine Similarity: Architecting an Enterprise Hybrid RAG Engine for SEC Filings
### How I combined BM25, FlashRank Cross-Encoders, and Ragas evaluation to eliminate hallucinations in complex corporate finance.

**By Muhammad Zainurrahman**  
*5 min read · Published in Towards AI / Level Up Coding*

---

<!-- IMAGE 1: HERO COVER BANNER -->
<!-- Place hero_cover.png here -->
<!-- Caption: The RAG-Ops Engine: Multi-Strategy Retrieval, Cross-Encoder Re-ranking, and Quantitative Evaluation over Corporate SEC Filings. -->

---

### Introduction: The Cost of a Hallucinated Decimal

In the high-stakes world of corporate finance, a hallucinated decimal point is not merely an embarrassing glitch — it is a multimillion-dollar compliance failure. 

When a quantitative analyst or portfolio manager asks:  
*"What was NVIDIA’s capital expenditure commitment for fiscal 2024, and how does it correlate with their TSMC foundry supply concentration?"*  

They are not seeking a plausible-sounding paragraph. They require the exact line-item dollar amount, the precise footnote clause, and zero creative storytelling.

Yet, the vast majority of real-world Retrieval-Augmented Generation (RAG) deployments in the wild remain stuck at the **"Naive RAG"** stage: chunk text every 500 tokens, generate dense vector embeddings, stash them in a vector database, and perform cosine similarity search.

When unleashed on SEC Form 10-K annual filings (dense, 100+ page audited reports packed with multi-column financial statements and nuanced legal footnotes), Naive RAG breaks down almost immediately.

I wanted to engineer a production-grade system that treats retrieval not as a casual semantic lookup, but as an auditable, high-precision engineering pipeline. 

The result is **RAG-Ops**: an enterprise comparative retrieval platform that benchmarks four competing RAG paradigms over official SEC filings (NVIDIA, Apple, Microsoft) and rigorously evaluates them using the **RAG Triad (Ragas / TruLens)**.

Here is the engineering journey of how I built it, the algorithmic trade-offs behind the pipeline, and the empirical benchmarks that prove why Hybrid Search + Cross-Encoder re-ranking is the true enterprise baseline.

---

### 1. The Business Problem: The 3 Fatal Flaws of Naive RAG

Before writing a line of code, I audited why standard vector search fails so miserably on audited financial filings. The failures boil down to three structural bottlenecks:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   3 CRITICAL NAIVE RAG FAILURE MODES                   │
├───────────────────┬───────────────────┬────────────────────────────────┤
│ 1. VOCABULARY     │ 2. CHUNK          │ 3. LOST IN THE                 │
│    MISMATCH       │    FRAGMENTATION  │    MIDDLE                      │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ Dense embeddings  │ Multi-row balance │ The generator LLM             │
│ blur exact digits │ sheets split.     │ ignores relevant context       │
│ and specific      │ Numbers lose      │ buried in the middle           │
│ accounting codes. │ column header     │ of long candidate              │
│                   │ associations.     │ context windows.               │
└───────────────────┴───────────────────┴────────────────────────────────┘
```

1. **The Vocabulary Mismatch:** Dense embedding models excel at broad conceptual semantics (*"customer satisfaction was positive"*), but struggle with exact lexical matches (*"ASC Topic 606"*, *"Form 10-K/A Item 8"*, or *"$1,680 million"*). A pure vector search often surfaces a general discussion on revenue while missing the exact balance sheet row.
2. **Chunk Fragmentation:** Arbitrary 500-token text splitters slice through financial statements mid-table. A number on line 32 loses its header on line 1, turning raw financial data into disconnected noise.
3. **The "Lost in the Middle" Phenomenon:** Pumping 20 raw candidate chunks into an LLM context window causes the model to over-index on the first and last chunks, completely ignoring crucial financial disclosures buried in chunks 8 through 12.

---

### 2. The Architecture: Multi-Strategy Comparative Pipeline

To systematically conquer these bottlenecks, I designed a modular, decoupled architecture supporting four distinct retrieval paradigms running against the same underlying corpus:

<!-- IMAGE 2: ARCHITECTURE FLOWCHART -->
<!-- Place architecture_diagram.png here -->
<!-- Caption: Multi-Strategy Retrieval Pipeline: Dual Sparse/Dense Engine, Reciprocal Rank Fusion, and CPU-Optimized Cross-Encoder Re-Ranking. -->

#### The 4 Paradigms Compared:
* **Paradigm 1: Naive RAG (Baseline)** — Recursive character chunking (512 tokens), dense embedding only, top-K cosine similarity.
* **Paradigm 2: Advanced Hybrid + FlashRank (The Standard)** — Parallel Sparse BM25 (exact keyword match) and Dense vector search, merged via Reciprocal Rank Fusion (RRF, $k=60$), then re-scored with a lightweight Cross-Encoder.
* **Paradigm 3: Query Transformation (HyDE)** — Generates a hypothetical financial paragraph via LLM first, using the hypothetical passage to search vector space (*Gao et al., 2022*).
* **Paradigm 4: Hierarchical Parent-Document** — Indexes granular child chunks (128 tokens) for needle-in-a-haystack search, but injects the full parent section (1024 tokens) into the LLM context prompt.

---

### 3. Engineering Implementation & The "Gotchas"

Building this pipeline revealed several critical engineering lessons that separates production RAG from tutorial prototypes:

#### A. Merging Apples and Oranges: Reciprocal Rank Fusion (RRF)
BM25 scores range from $0$ to $\infty$, while Cosine Similarity outputs values between $-1.0$ and $+1.0$. Attempting a weighted linear sum ($0.5 \times \text{BM25} + 0.5 \times \text{Cosine}$) leads to calibration nightmares whenever a query matches a rare token.

Instead, I implemented **Reciprocal Rank Fusion (RRF)** with standard industrial smoothing ($k=60$):

$$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

```python
def reciprocal_rank_fusion(
    bm25_results: list[dict], 
    dense_results: list[dict], 
    k: int = 60
) -> list[dict]:
    """Combines heterogeneous rank lists into a single calibrated ranking."""
    fused_scores = {}
    
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

#### B. Eliminating GPU Overhead: FlashRank Cross-Encoder
Traditional Cross-Encoders (like `cross-encoder/ms-marco-MiniLM-L-6-v2`) require heavy GPU instances or introduce 200ms+ inference latencies. 

I integrated **FlashRank** — a 4-bit quantized, miniaturized cross-encoder optimized for CPU execution. It processes 20 candidate chunks in **under 20ms**, evaluating full all-to-all cross-attention between the query and each candidate passage before discarding any passage scoring below a strict relevance threshold ($\tau = 0.35$).

---

### 4. From Math to Money: The Empirical Benchmark

To quantitatively validate system performance rather than relying on anecdotal "vibes," I curated a **Golden Test Set of 50 verified financial questions** and evaluated the outputs across the **RAG Triad (Ragas / TruLens standard)**:

<!-- IMAGE 3: RADAR BENCHMARK CHART -->
<!-- Place radar_benchmark_chart.png here -->
<!-- Caption: Quantitative Benchmark across 5 Core Dimensions: Faithfulness, Context Precision, Context Recall, Answer Relevance, and Latency. -->

#### Benchmark Results (50 Golden Financial QA Pairs):

| Metric / Architecture | 1. Naive RAG | 2. Hybrid + FlashRank | 3. HyDE | 4. Hierarchical Parent |
| :--- | :---: | :---: | :---: | :---: |
| **Faithfulness (Anti-Hallucination)** | 0.68 | **0.96** | 0.89 | 0.94 |
| **Context Precision** | 0.52 | **0.89** | 0.74 | 0.82 |
| **Context Recall** | 0.61 | **0.93** | 0.81 | 0.91 |
| **Answer Relevance** | 0.71 | **0.95** | 0.92 | 0.93 |
| **MRR @ 5** | 0.48 | **0.87** | 0.72 | 0.84 |
| **Retrieval P95 Latency** | **12 ms** | 38 ms | 145 ms | 28 ms |

#### The Verdict:
* **Hybrid + Re-Ranking is the Uncontested Champion:** It delivered a **+37% boost in Context Precision** and improved **MRR@5 from 0.48 to 0.87**, while adding only 26ms of CPU overhead.
* **Faithfulness Reached 0.96:** Factual hallucinations on corporate metrics were almost completely eliminated because the Cross-Encoder aggressively prunes noisy or tangential passages before context synthesis.

---

### 5. Production Deployment: The Interactive A/B Arena

Rather than keeping this system confined to a terminal script, I deployed the complete architecture to production:
* **Backend:** FastAPI microservice deployed on **Render** with dynamic CORS and streaming response endpoints.
* **Frontend:** Interactive **React (Vite) + Tailwind CSS** dashboard hosted on **Vercel**.

<!-- IMAGE 4: LIVE DASHBOARD A/B ARENA SCREENSHOT -->
<!-- Place ab_arena_screenshot.png here -->
<!-- Caption: The Interactive A/B Arena: Live split-screen comparison revealing exact context differences between Naive RAG and Hybrid + Re-Ranker. -->

Visitors can test any financial query in the **A/B Arena** to watch side-by-side execution:
1. **The Left Screen (Naive RAG)** grabs broad, noisy chunks that fail to answer the specific question.
2. **The Right Screen (Hybrid + FlashRank)** pinpoints the exact table row and highlights the rank movement of candidate chunks in real time.

---

### Conclusion: Key Takeaways for AI Engineers

Building RAG-Ops reinforced a fundamental truth in applied AI: **better models rarely fix bad context.** Before reaching for a larger LLM or expensive fine-tuning runs, engineer the retrieval surface first.

* **Hybrid Search (BM25 + Dense) is mandatory** whenever queries contain entities, SKU numbers, or financial metrics.
* **Cross-Encoder re-ranking provides the highest ROI** in the entire RAG pipeline — turning a noisy top-20 candidate pool into a pristine top-3 context window.
* **Evaluation-Driven Development (EDD) is essential:** You cannot improve what you cannot measure; tracking the RAG Triad (Faithfulness, Precision, Recall) is what separates production engineering from weekend hobby projects.

---

### Technical Stack:
* **Retrieval & Reranking:** Sparse BM25, Dense Vector Cosine Similarity, Reciprocal Rank Fusion ($k=60$), FlashRank Cross-Encoder
* **Evaluation Suite:** Ragas / TruLens (Faithfulness, Context Precision, Context Recall, Answer Relevance, MRR@5)
* **Backend:** FastAPI, Python 3.11, Pydantic v2, Docker
* **Frontend:** React 18, Vite, Tailwind CSS, Lucide Icons, Recharts
* **Cloud Infrastructure:** Vercel (Edge UI), Render (API Gateway), DeepSeek & Gemini via OpenRouter

---

### Credits & Continuity

* 🌐 **Test the Live Dashboard:** [ragops-benchmark.vercel.app](https://ragops-benchmark.vercel.app/)
* 📦 **Explore the Full Source Code on GitHub:** [github.com/MuhZainur/ragops-benchmark](https://github.com/MuhZainur/ragops-benchmark)
* 💼 **Connect with me on LinkedIn:** [linkedin.com/in/muhammad-zainurrahman](https://www.linkedin.com/in/muhammad-zainurrahman)

*Tags: `#MachineLearning` `#ArtificialIntelligence` `#DataScience` `#LLM` `#MLOps`*
