<div align="center">

<img src="logo.png" alt="TalentLens AI" width="300"/>

# TalentLens AI

### An AI recruiter that understands candidates — not just keywords

*Built for the [Redrob × H2S Data & AI Challenge](https://hack2skill.com)*

---

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](YOUR_STREAMLIT_URL_HERE)
[![BGE Embeddings](https://img.shields.io/badge/BGE-Dense%20Embeddings-6366F1?style=for-the-badge)](https://huggingface.co/BAAI/bge-base-en-v1.5)
[![Runtime](https://img.shields.io/badge/Runtime-~80s%20on%20CPU-22C55E?style=for-the-badge)]()
[![Honeypots](https://img.shields.io/badge/Honeypots%20in%20Top%20100-0%25-22C55E?style=for-the-badge)]()

[🚀 Live Demo](YOUR_STREAMLIT_URL_HERE) · [📊 View Submission](outputs/submission.xlsx) · [🧪 Run Tests](#tests)

</div>

---

## What is TalentLens AI?

TalentLens AI ranks **100,000 candidates** against a job description the way a great recruiter would — by understanding who **genuinely fits the role**, not by matching keywords.

Most systems do this:
```
Candidate has "Python" + "FAISS" + "Vector DB"  →  80% match
```

TalentLens does this:
```
Where did they actually USE these skills? (career history, not just skill list)
Did their seniority grow over time? (career trajectory)
Are they actually available and engaged? (behavioral signals)
Is this profile real? (honeypot integrity detection)
```

---

## Live Demo

<div align="center">

### 🚀 [Open TalentLens AI →](YOUR_STREAMLIT_URL_HERE)

*Tick "Use dataset files from data/ folder" → click **Run Scoring Pipeline***

</div>

---

## Dashboard

### Recruiter Leaderboard
> Ranked candidates with **Strength** signals in green and **Concerns** in amber — every decision explained.

![Leaderboard](screenshots/leaderboard.png)

---

### Talent Pool Analytics
> Experience band distribution and location segmentation across all 100,000 candidates.

![Analytics](screenshots/analytics.png)

---

### Job Requirements Config & AI Guardrails
> JD-parsed requirements and system exclusion rules — fully transparent scoring rules.

![JD Config](screenshots/jd_config.png)

---

### Pipeline Calibration
> Live weight tuning — adjust scoring components and see the leaderboard rerank in real time.

![Calibration](screenshots/calibration.png)

---

### Candidate Comparison
> Side-by-side breakdown of any two candidates across all scoring dimensions.

![Compare](screenshots/compare.png)

---

## Results

<div align="center">

| Metric | Value |
|:---|:---:|
| Candidates scored | **100,000** |
| Wall-clock runtime | **~80s** (budget: 300s) |
| Memory usage | **< 8 GB** (budget: 16 GB) |
| Network calls during ranking | **0** |
| Honeypots in top 100 | **0 (0.0%)** |
| Top candidate score | **83.8%** |

</div>

### Top 10 Ranked Candidates

| # | Role | Company | Score | Key Signals |
|:---:|:---|:---|:---:|:---|
| 1 | Search Engineer | Sarvam AI | 83.8% | 7.6 yrs · Gurgaon · Embeddings + Vector DB · 94% response |
| 2 | Sr. ML Engineer | Genpact AI | 82.6% | 6.1 yrs · Pune · Embeddings + Vector DB + Ranking Eval |
| 3 | Sr. ML Engineer | Zomato | 81.2% | 7.2 yrs · Noida · 15d notice · Open to work |
| 4 | Rec. Systems Eng. | Wysa | 80.4% | 7.9 yrs · Noida · Embeddings + Vector DB + Eval |
| 5 | NLP Engineer | Haptik | 75.1% | 6.5 yrs · Pune · 15d notice · Full skill match |
| 6 | ML Engineer | Swiggy | 74.3% | 7.1 yrs · Bangalore · Embeddings + Python |
| 7 | Search Eng. | Meesho | 73.8% | 6.8 yrs · Bangalore · Vector DB + Ranking Eval |
| 8 | Applied Scientist | Razorpay | 72.9% | 5.9 yrs · Bangalore · Embeddings + Python |
| 9 | ML Engineer | PhonePe | 71.4% | 6.3 yrs · Bangalore · Vector DB |
| 10 | Senior Data Scientist | Flipkart | 70.2% | 7.5 yrs · Bangalore · Retrieval + Python |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JD + 100,000 Candidates                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feature Extraction                        │
│  Career history · Skills · Education · Location · Signals   │
└────────────┬──────────────────────────────┬────────────────┘
             │                              │
             ▼                              ▼
┌────────────────────────┐    ┌─────────────────────────────┐
│   Must-Have Skills 55% │    │     Semantic Fit 45%        │
│                        │    │                             │
│  Career-history terms  │    │  BGE Dense Embeddings       │
│  weighted 3× over      │    │  (or TF-IDF+LSA fallback)   │
│  skill-list terms      │    │                             │
└────────────┬───────────┘    └──────────────┬──────────────┘
             │                               │
             └──────────────┬────────────────┘
                            │
                            ▼ CORE SKILL FIT  (the gate)
                            │
               ┌────────────┼────────────────┐
               │            │                │
               ▼            ▼                ▼
        × Experience   × Location      × Disqualifier
          [0.50–1.0]   [0.55–1.0]        [0.30–1.0]
          multiplier   multiplier         multiplier
               │                          │
               │    × Behavioral          │
               │      [0.40–1.10]         │
               │      multiplier          │
               │                          │
               │    × Honeypot            │
               │      [0.01 or 1.0]       │
               └──────────┬───────────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │  Top 100 Ranked CSV  │
               │  id · rank · score   │
               │  · reasoning         │
               └──────────────────────┘
```

---

## Scoring Components

| Component | Weight | Method | Why It Matters |
|:---|:---:|:---|:---|
| **Must-Have Skills** | 55% | Regex + duration depth check | Career-history mentions weighted **3×** over skill-list tags — catches keyword stuffers |
| **Semantic Fit** | 45% | BGE cosine / TF-IDF+LSA | Finds "right skills, wrong buzzwords" candidates that pure keyword search misses |
| **Experience Multiplier** | ×mod | Triangular fn, peak 6-8 yrs | Soft floor (0.5×) so strong-skill candidates outside the band still pass |
| **Location Multiplier** | ×mod | Geography check | Target India cities = 1.0×  ·  Willing to relocate = 0.75×  ·  Global = 0.55× |
| **Disqualifier Guard** | ×mod | Rule-based | Consulting-only career · pure research · CV/Speech without NLP · senior title + no code signal |
| **Behavioral Signal** | ×mod | redrob_signals | Response rate · last active · open-to-work · notice period · verification |
| **Honeypot Shield** | ×mod | 7-point anomaly detector | 0.01× if ≥2 checks fire · 1.0× otherwise |

### Final Score Formula

```
final_score = core_skill_fit
            × experience_multiplier
            × location_multiplier  
            × disqualifier_multiplier
            × behavioral_multiplier
            × honeypot_multiplier
```

> Multiplicative, not additive — a serious red flag collapses the score to near-zero regardless of skill fit. An additive model would let a high base score paper it over.

---

## Key Design Decisions

<details>
<summary><b>Why multiplicative scoring?</b></summary>

A Civil Engineer in Gurgaon with 7 years experience should not outscore an NLP Engineer in London just because "location + experience adds as much to the sum as skills do." Skills are the gate. Location and experience modulate, they don't compete.

</details>

<details>
<summary><b>Why weight career history 3× over skill lists?</b></summary>

A skill appearing in an actual job description (what the person built and shipped) is far harder to fake than adding "Pinecone" to a skills list. This directly counters the "Data Engineer listing 15 advanced ML skills that never appear in their career history" pattern — the exact trap the JD calls out.

</details>

<details>
<summary><b>Why BGE embeddings + TF-IDF fallback?</b></summary>

BGE dense vectors catch the "right experience, wrong buzzwords" candidate that keyword matching misses. TF-IDF + LSA is the CPU-only fallback when no precomputed cache is available — both satisfy the no-network-calls-at-ranking-time constraint by running entirely offline.

</details>

<details>
<summary><b>Why 7 independent honeypot checks?</b></summary>

Requiring 2+ independent anomaly checks to fire keeps false-positive rate low on genuinely unusual-but-real profiles, while reliably catching synthetically constructed honeypots that fail multiple checks simultaneously. Checks include: expert skills with zero duration, overlapping employment dates, endorsements exceeding connections ×4, unsupported seniority jumps, and more.

</details>

<details>
<summary><b>Why deterministic reasoning generation?</b></summary>

Every fact in the reasoning column is pulled directly from parsed candidate data — no LLM hallucination risk, no network dependency during the graded run. The optional `scripts/ai_reasoning_polish.py` uses Ollama to write better prose around those same verified facts — but falls back safely if Ollama is unavailable, so the submission never breaks.

</details>

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/candidate-ranking-ai.git
cd candidate-ranking-ai
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Place data (not committed — 487MB)
# copy candidates.jsonl → data/candidates.jsonl

# 3. Run
python rank.py

# 4. Validate
python scripts/validate_submission.py outputs/submission.xlsx


### Run the Streamlit App

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Tests

```bash
python -m pytest tests/ -v


## Project Structure

```
candidate-ranking-ai/
│
├── rank.py                         # ← single entrypoint — run this
├── app.py                          # Streamlit workspace (5 tabs)
├── requirements.txt
├── logo.png
│
├── src/
│   ├── config.py                   # all weights, constants, JD-derived rules
│   ├── data_loader.py              # .jsonl and .jsonl.gz support
│   ├── feature_extraction.py       # raw JSON → scoring-ready feature dict
│   ├── reasoning.py                # deterministic fact-grounded reasoning
│   └── scoring/
│       ├── composite.py            # multiplies all components into final score
│       ├── must_have_skills.py     # embeddings / vector-DB / eval / Python
│       ├── semantic_fit.py         # TF-IDF + LSA (CPU semantic engine)
│       ├── hard_filters.py         # experience, location, JD disqualifiers
│       ├── behavioral_signal.py    # redrob_signals multiplier
│       └── honeypot_detection.py   # 7-point integrity check
│
├── scripts/
│   ├── validate_submission.py      # official hackathon validator (unchanged)
│   ├── precompute_embeddings.py    # GPU: generate BGE cache once
│   ├── csv_to_xlsx.py              # convert validated CSV → portal XLSX
│   ├── ai_reasoning_polish.py      # optional: Ollama prose polish
│   └── ollama_rerank.py            # optional: Ollama reranker on top 200
│
├── eval/
│   └── evaluate.py                 # private gold-set NDCG/MAP harness
│
├── tests/
│   └── test_pipeline.py            # 19 unit tests (all passing)
│
├── data/
│   ├── job_description.md          # JD (committed)
│   ├── sample_candidates.json      # 50-candidate smoke test sample
│
└── outputs/
    └── submission.xlsx             # final ranked output (portal-ready)
```

---

## Hackathon Scoring Formula

```
hackathon_score = 0.50 × NDCG@10
                + 0.30 × NDCG@50
                + 0.15 × MAP
                + 0.05 × P@10
```

**80% of the score lives in the top 50 picks.** TalentLens optimizes hard for precision at the top of the ranking, which is where recruiter trust is won or lost.


---

<div align="center">

**TalentLens AI** &nbsp;·&nbsp; Redrob × H2S Data & AI Challenge &nbsp;·&nbsp; 2026

*CPU + GPU inference paths &nbsp;·&nbsp; Zero network calls during ranking &nbsp;·&nbsp; 19/19 tests passing*

</div>
