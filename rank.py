
"""
rank.py — produce the top-100 candidate ranking CSV.

Usage (baseline, CPU-only, no network):
    python rank.py

Usage (with precomputed BGE embeddings for better semantic fit):
    python rank.py --embedding-cache data/embeddings_cache.npz

Must run in <=5 min, <=16GB RAM, CPU only, zero network calls. This file
makes none; if --embedding-cache is used, the .npz was generated offline.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.data_loader import load_candidates, load_job_description
from src.feature_extraction import extract_features
from src.reasoning import generate_reasoning
from src.scoring import composite
from src.scoring.semantic_fit import fit_semantic_space, score_semantic_fit_batch


def parse_args():
    p = argparse.ArgumentParser(description="Rank candidates against the Redrob JD.")
    p.add_argument("--candidates", type=Path, default=config.CANDIDATES_PATH)
    p.add_argument("--jd",        type=Path, default=config.JD_PATH)
    p.add_argument("--out",       type=Path, default=config.OUTPUT_DIR / "submission.csv")
    p.add_argument("--top-k",     type=int,  default=config.TOP_K)
    p.add_argument(
        "--embedding-cache", type=Path, default=None,
        help="Path to .npz from scripts/precompute_embeddings.py. "
             "Uses dense BGE cosine scores instead of TF-IDF/LSA.",
    )
    return p.parse_args()


def _semantic_from_cache(cache_path: Path, candidate_ids: list) -> np.ndarray:
    """Load precomputed BGE embeddings → cosine similarity scores, ordered by candidate_ids."""
    data   = np.load(cache_path, allow_pickle=True)
    jd_emb = data["jd_embedding"]            # (1, dim)
    c_emb  = data["candidate_embeddings"]    # (N, dim)
    cached = list(data["candidate_ids"])
    id2row = {cid: i for i, cid in enumerate(cached)}
    rows   = [id2row[cid] for cid in candidate_ids]
    sims   = (c_emb[rows] @ jd_emb.T).squeeze()   # dot == cosine (L2-normed)
    return (sims + 1.0) / 2.0


def main():
    args = parse_args()
    t0   = time.time()

    print(f"[1/5] Loading candidates from {args.candidates} ...", flush=True)
    candidates = load_candidates(args.candidates)
    print(f"      {len(candidates):,} candidates  ({time.time()-t0:.1f}s)", flush=True)

    print(f"[2/5] Loading JD from {args.jd} ...", flush=True)
    jd_text = load_job_description(args.jd)

    print("[3/5] Extracting features ...", flush=True)
    t1 = time.time()
    features_list = [extract_features(c) for c in candidates]
    print(f"      Done  ({time.time()-t1:.1f}s)", flush=True)

    t2 = time.time()
    use_cache = args.embedding_cache and Path(args.embedding_cache).exists()
    if use_cache:
        print(f"[4/5] Loading BGE embedding cache {args.embedding_cache} ...", flush=True)
        cids = [f["candidate_id"] for f in features_list]
        semantic_scores = _semantic_from_cache(args.embedding_cache, cids)
    else:
        print("[4/5] TF-IDF + LSA semantic fit (CPU) ...", flush=True)
        texts = [f["full_text"] for f in features_list]
        _, _, jd_vec, c_vecs = fit_semantic_space(jd_text, texts)
        semantic_scores = score_semantic_fit_batch(jd_vec, c_vecs)
    print(f"      Done  ({time.time()-t2:.1f}s)", flush=True)

    print(f"[5/5] Scoring all candidates, picking top {args.top_k} ...", flush=True)
    t3 = time.time()
    results = [
        composite.score_candidate(feats, float(sem))
        for feats, sem in zip(features_list, semantic_scores)
    ]
    print(f"      Done  ({time.time()-t3:.1f}s)", flush=True)

    # sort: descending score (rounded to 6 decimal places to match CSV precision), then ascending candidate_id for ties
    results.sort(key=lambda r: (-round(r["final_score"], 6), r["candidate_id"]))
    top = results[: args.top_k]
    feats_by_id = {f["candidate_id"]: f for f in features_list}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, r in enumerate(top, start=1):
            feats = feats_by_id[r["candidate_id"]]
            writer.writerow([
                r["candidate_id"], i,
                f"{r['final_score']:.6f}",
                generate_reasoning(feats, r, i),
            ])

    n_hp  = sum(1 for r in top if r["is_honeypot_flagged"])
    total = time.time() - t0
    print(f"\nWrote {args.out}  ({len(top)} rows)")
    print(f"Honeypots in top {args.top_k}: {n_hp}  ({n_hp/args.top_k:.1%})  [limit: <10%]")
    print(f"Wall-clock: {total:.1f}s  (budget: 300s)")
    if total > 300:
        print("WARNING: exceeded compute budget!", file=sys.stderr)


if __name__ == "__main__":
    main()
