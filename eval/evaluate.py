#!/usr/bin/env python3
"""
eval/evaluate.py

Private evaluation harness: build your own gold-set labels and compute
NDCG@10, NDCG@50, MAP, and P@10 — the exact metrics used by the hackathon
judges — so you can tune scoring weights without burning your 3-submission cap.

Step 1 — generate a label template for candidates you want to hand-label:
    python eval/evaluate.py --gen-labels --candidates data/candidates.jsonl \
        --jd data/job_description.md --sample 150 --out eval/gold_labels.csv

Step 2 — open eval/gold_labels.csv, fill in the 'relevance' column:
    3 = strong fit (would definitely interview)
    2 = moderate fit (would phone screen)
    1 = weak fit (interesting but not a match)
    0 = not relevant / honeypot / disqualifier

Step 3 — evaluate your submission:
    python eval/evaluate.py --submission outputs/submission.csv \
        --labels eval/gold_labels.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_candidates, load_job_description
from src.feature_extraction import extract_features


# ── Gold-label generation ────────────────────────────────────────────────────

def _candidate_summary(c: dict) -> str:
    p = c.get("profile", {})
    skills = ", ".join(s["name"] for s in (c.get("skills") or [])[:6])
    return (
        f"{p.get('current_title','?')} @ {p.get('current_company','?')} | "
        f"{p.get('years_of_experience','?')} yrs | {p.get('location','?')} | "
        f"Skills: {skills}"
    )


def gen_labels(candidates_path, jd_path, sample_n, out_path):
    """
    Sample candidates that are plausibly in the top 200, plus some negatives,
    and write a CSV for hand-labelling.
    """
    from src.scoring.semantic_fit import fit_semantic_space, score_semantic_fit_batch

    print("Loading data for label generation...")
    candidates = load_candidates(candidates_path)
    jd_text    = load_job_description(jd_path)
    features   = [extract_features(c) for c in candidates]
    texts      = [f["full_text"] for f in features]

    print("Fitting semantic space to find candidates worth labelling...")
    _, _, jd_vec, c_vecs = fit_semantic_space(jd_text, texts)
    sims = score_semantic_fit_batch(jd_vec, c_vecs)

    # grab top-300 by semantic score + 50 random negatives
    order   = sorted(range(len(sims)), key=lambda i: -sims[i])
    indices = order[:300] + order[-50:]
    # deduplicate while preserving order
    seen, unique = set(), []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    unique = unique[: sample_n]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "semantic_sim", "summary", "relevance"])
        for i in unique:
            writer.writerow([
                features[i]["candidate_id"],
                f"{sims[i]:.4f}",
                _candidate_summary(candidates[i]),
                "",   # ← you fill this in: 0/1/2/3
            ])
    print(f"Wrote {len(unique)} candidates to {out_path}")
    print("Fill in the 'relevance' column (0/1/2/3) and re-run with --submission.")


# ── Metric computation ───────────────────────────────────────────────────────

def _dcg(relevances: list[float], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += (2**rel - 1) / math.log2(i + 1)
    return total


def _ndcg(predicted_ids: list[str], gold: dict[str, int], k: int) -> float:
    rels    = [gold.get(cid, 0) for cid in predicted_ids[:k]]
    ideal   = sorted(gold.values(), reverse=True)
    dcg_val = _dcg(rels, k)
    idcg    = _dcg(ideal, k)
    return dcg_val / idcg if idcg > 0 else 0.0


def _average_precision(predicted_ids: list[str], gold: dict[str, int], k: int) -> float:
    hits, precision_sum = 0, 0.0
    relevant = {cid for cid, rel in gold.items() if rel >= 2}
    for i, cid in enumerate(predicted_ids[:k], start=1):
        if cid in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant) if relevant else 0.0


def evaluate(submission_path, labels_path):
    gold: dict[str, int] = {}
    with open(labels_path) as f:
        for row in csv.DictReader(f):
            rel = row.get("relevance", "").strip()
            if rel in ("0", "1", "2", "3"):
                gold[row["candidate_id"]] = int(rel)

    if not gold:
        print("ERROR: No filled-in labels found. Fill the 'relevance' column first.")
        sys.exit(1)

    predicted = []
    with open(submission_path) as f:
        for row in csv.DictReader(f):
            predicted.append(row["candidate_id"])

    ndcg10 = _ndcg(predicted, gold, 10)
    ndcg50 = _ndcg(predicted, gold, 50)
    map_   = _average_precision(predicted, gold, 100)
    p10    = sum(1 for cid in predicted[:10] if gold.get(cid, 0) >= 2) / 10

    # Hackathon formula
    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_ + 0.05 * p10

    print(f"\n{'Metric':<15} {'Score':>8}")
    print("-" * 25)
    print(f"{'NDCG@10':<15} {ndcg10:>8.4f}   (weight: 50%)")
    print(f"{'NDCG@50':<15} {ndcg50:>8.4f}   (weight: 30%)")
    print(f"{'MAP':<15} {map_:>8.4f}   (weight: 15%)")
    print(f"{'P@10':<15} {p10:>8.4f}   (weight:  5%)")
    print("-" * 25)
    print(f"{'COMPOSITE':<15} {composite:>8.4f}")
    print(f"\n({len(gold)} labelled candidates, {sum(1 for v in gold.values() if v >= 2)} relevant)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    gen = sub.add_parser("gen-labels", help="Generate label template CSV")
    gen.add_argument("--candidates", type=Path, default=Path("data/candidates.jsonl"))
    gen.add_argument("--jd",         type=Path, default=Path("data/job_description.md"))
    gen.add_argument("--sample",     type=int,  default=150)
    gen.add_argument("--out",        type=Path, default=Path("eval/gold_labels.csv"))

    ev = sub.add_parser("score", help="Evaluate a submission CSV")
    ev.add_argument("--submission", type=Path, required=True)
    ev.add_argument("--labels",     type=Path, default=Path("eval/gold_labels.csv"))

    # allow flat args too (backwards compat)
    p.add_argument("--gen-labels",  action="store_true")
    p.add_argument("--candidates",  type=Path, default=Path("data/candidates.jsonl"))
    p.add_argument("--jd",          type=Path, default=Path("data/job_description.md"))
    p.add_argument("--sample",      type=int,  default=150)
    p.add_argument("--out",         type=Path, default=Path("eval/gold_labels.csv"))
    p.add_argument("--submission",  type=Path)
    p.add_argument("--labels",      type=Path, default=Path("eval/gold_labels.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if getattr(args, "gen_labels", False) or args.cmd == "gen-labels":
        gen_labels(args.candidates, args.jd, args.sample, args.out)
    elif args.cmd == "score" or args.submission:
        evaluate(args.submission, args.labels)
    else:
        print("Use --gen-labels or --submission. See --help.")
