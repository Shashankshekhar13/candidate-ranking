# TalentLens AI — Redrob × H2S Data & AI Challenge

An AI recruiter that understands candidates, explains every decision, catches
suspicious profiles, and finds the best fit through semantic reasoning — not
keyword matching.

## Reproduce the submission in one command

```bash
python rank.py \
  --candidates data/candidates.jsonl \
  --jd data/job_description.md \
  --out outputs/submission.csv
```

Runtime: ~80s on CPU, well under the 300s/16GB/no-network budget.
Validates cleanly with `python scripts/validate_submission.py outputs/submission.csv`.

## Architecture

```
JD + 100k candidates
        │
        ▼
  Feature extraction      (src/feature_extraction.py)
  Career, skills, signals, location, dates
        │
        ├──▶ Core Skill Fit (gate, 100% of base score)
        │         Must-have skills (55%)     ← career-history weighted
        │         Semantic fit / LSA (45%)   ← TF-IDF + TruncatedSVD
        │
        ├──▶ × Experience multiplier  [0.5, 1.0]
        ├──▶ × Location multiplier    [0.55, 1.0]
        ├──▶ × Disqualifier multiplier (JD red flags)
        ├──▶ × Behavioral multiplier  (engagement signals)
        └──▶ × Honeypot multiplier    (anomaly detection)
                │
                ▼
          Top 100 CSV (candidate_id, rank, score, reasoning)
```

### Design choices

**Skills as the gate, not an additive term.**
Location and years-of-experience are JD-described secondary factors ("flexible",
"a range, not a requirement"). A Civil Engineer in Gurgaon with 7 years experience
should not score above an NLP Engineer in London just because location/experience
add as much to the sum as skills do. So location and experience are multiplicative
*modifiers* on the skill-fit core, not competing additive components.

**Career-history weighted skill matching.**
A skill appearing in an actual job description (what the person worked on) is
weighted 3× over the same term appearing only in the skills list (easy to fake).
This directly counters the "Data Engineer listing 15 advanced ML skills none of
which appear in their career history" honeypot/keyword-stuffer pattern.

**TF-IDF + LSA semantic fit (not a neural model).**
Captures topical overlap (recommendation systems, retrieval, ranking) without
requiring model weights to be downloaded at runtime — satisfying the no-network
constraint by construction. Runs in ~55s on CPU for 100k candidates.

**Multiplicative combination.**
`final_score = core_fit × exp_mult × loc_mult × disqualifier_mult × behavioral_mult × honeypot_mult`
A multiplicative model means a serious red flag (honeypot, pure-consulting career)
collapses the score; an additive model would let a strong base score paper it over.

**Deterministic reasoning generator.**
Every fact in the reasoning column is pulled directly from the candidate's parsed
data — no LLM call, no hallucination risk, no network required. Variation comes
from rotating sentence templates chosen deterministically per candidate_id.

## Project structure

```
candidate-ranking-ai/
├── rank.py                      # single entrypoint — run this
├── requirements.txt
├── src/
│   ├── config.py                # all weights, constants, JD-derived rules
│   ├── data_loader.py
│   ├── feature_extraction.py
│   ├── reasoning.py
│   └── scoring/
│       ├── hard_filters.py      # experience band, location, JD disqualifiers
│       ├── must_have_skills.py  # production embeddings/vector-DB/eval/Python
│       ├── semantic_fit.py      # TF-IDF + LSA
│       ├── behavioral_signal.py # redrob_signals engagement multiplier
│       ├── honeypot_detection.py
│       └── composite.py         # combines all components
├── scripts/
│   └── validate_submission.py  # official hackathon validator (unchanged)
├── data/
│   ├── candidates.jsonl
│   └── job_description.md
└── outputs/
    └── submission.csv
```

## Upgrading to GPU embeddings (Version 2)

If you have a GPU (RTX 3050 or better), swap the semantic fit component for
`sentence-transformers/BAAI/bge-large-en-v1.5`. This should improve NDCG@10
meaningfully for the "right experience, wrong keywords" candidates the JD cares about.

**Critical constraint**: the submission validator runs with no network. You must
**precompute and cache** the embeddings:

```bash
# One-time precompute (can use network, takes ~10 min on RTX 3050)
python scripts/precompute_embeddings.py \
  --candidates data/candidates.jsonl \
  --jd data/job_description.md \
  --out data/embeddings_cache.npz

# rank.py then loads from cache, no network needed
python rank.py --use-embedding-cache data/embeddings_cache.npz
```

## Upgrading to Ollama reranker (Version 3)

Pull a small local model for reranking only the top 200 (not all 100k):

```bash
ollama pull qwen2.5:7b
python scripts/rerank_top200.py --input outputs/submission.csv --out outputs/submission_reranked.csv
```

This is a dev-only step — the reranker output is a static CSV that gets submitted,
so no model runs at validation time.

## Scoring formula reference

Per `submission_spec.docx`:
`final_score = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10`

80% of the score lives in the top 50 picks. Spend calibration effort there, not on rank 90-100.

## AI tool disclosure

This solution was developed with AI-assisted code generation. All code has been
reviewed, understood, and validated end-to-end by the submitting team.
