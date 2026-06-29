"""
Fast embedding precomputation — manual GPU batching for maximum speed.
Tokenizes ALL texts upfront (CPU), then encodes batches directly on GPU.
"""

import argparse
import json
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from transformers import AutoModel, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, default=Path("data/candidates.jsonl"))
    p.add_argument("--jd", type=Path, default=Path("data/job_description.md"))
    p.add_argument("--out", type=Path, default=Path("data/embeddings_cache.npz"))
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=512)
    return p.parse_args()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_texts(texts, tokenizer, model, batch_size, max_length, device, desc="Encoding"):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # Move entire batch to GPU at once
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        # CLS token pooling (best for BGE models)
        embeddings = output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu().float().numpy())

    return np.vstack(all_embeddings)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer + model directly via transformers (bypasses sentence-transformers overhead)
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    # Enable FP16 for faster GPU inference
    if device.type == "cuda":
        model = model.half()
        print("FP16 enabled")

    # Load candidates
    print(f"Loading candidates from {args.candidates}...")
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates")

    # Extract texts directly (no feature_extraction overhead)
    print("Extracting texts...")
    candidate_ids = []
    candidate_texts = []

    for c in candidates:
        profile = c.get("profile", {}) or {}
        career = c.get("career_history", []) or []
        skills = c.get("skills", []) or []

        headline = profile.get("headline", "") or ""
        summary = (profile.get("summary", "") or "")[:300]
        titles = " ".join(c.get("title", "") or "" for c in career[:5])
        skill_names = " ".join(s.get("name", "") or "" for s in skills[:20])
        career_text = " ".join(
            f"{c.get('title','')} {c.get('company','')} {c.get('description','')[:100]}"
            for c in career[:4]
        )

        full_text = f"{headline} {summary} {titles} {skill_names} {career_text}"[:1500]

        candidate_ids.append(c.get("candidate_id"))
        candidate_texts.append(full_text)

    # Load JD
    jd_text = Path(args.jd).read_text(encoding="utf-8")
    is_bge = "bge" in args.model.lower()
    jd_query = f"Represent this sentence for searching relevant passages: {jd_text}" if is_bge else jd_text

    # Encode JD
    print("Encoding JD...")
    jd_embedding = encode_texts(
        [jd_query], tokenizer, model, batch_size=1,
        max_length=args.max_length, device=device, desc="JD"
    )

    # Encode candidates
    print(f"Encoding {len(candidate_texts)} candidates in batches of {args.batch_size}...")
    candidate_embeddings = encode_texts(
        candidate_texts, tokenizer, model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        desc="Candidates",
    )

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        jd_embedding=jd_embedding,
        candidate_embeddings=candidate_embeddings,
        candidate_ids=np.array(candidate_ids),
    )

    print(f"\nSaved to {args.out}")
    print(f"  JD shape:         {jd_embedding.shape}")
    print(f"  Candidates shape: {candidate_embeddings.shape}")
    print(f"\nNow run: python rank.py --embedding-cache {args.out}")


if __name__ == "__main__":
    main()