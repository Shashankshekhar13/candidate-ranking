import argparse
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_candidates, load_job_description

RERANK_PROMPT = """\
You are an expert technical recruiter. Below is a job description and a candidate profile.
Score this candidate's fit for the role from 0.0 (no fit) to 1.0 (perfect fit).

Consider:
1. Production experience with embeddings, vector databases, semantic search
2. Years of experience and seniority band (5-9 years ideal)
3. Career trajectory — are they growing into ML/AI or away from it?
4. Behavioral signals: engagement, availability, response rate

IMPORTANT: Respond with ONLY a JSON object in this exact format:
{"score": <float 0.0-1.0>, "reason": "<one sentence>"}

=== JOB DESCRIPTION ===
{jd}

=== CANDIDATE PROFILE ===
ID: {candidate_id}
Title: {title}
Company: {company}
Years Experience: {yoe}
Location: {location}
Summary: {summary}
Top Skills: {skills}
Career: {career}
"""


def _build_profile_text(candidate: dict) -> dict:
    p = candidate.get("profile", {})
    skills = ", ".join(s["name"] for s in (candidate.get("skills") or [])[:8])
    career_parts = []
    for role in (candidate.get("career_history") or [])[:3]:
        career_parts.append(
            f"{role.get('title','?')} at {role.get('company','?')} "
            f"({role.get('duration_months', '?')} months): {(role.get('description') or '')[:150]}"
        )
    return {
        "candidate_id": candidate.get("candidate_id"),
        "title":   p.get("current_title", ""),
        "company": p.get("current_company", ""),
        "yoe":     p.get("years_of_experience", ""),
        "location":p.get("location", ""),
        "summary": (p.get("summary") or "")[:300],
        "skills":  skills,
        "career":  " | ".join(career_parts),
    }


def _query_ollama(model: str, prompt: str) -> dict | None:
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        text = response["message"]["content"].strip()
        # strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        print(f"  Ollama error: {e}", file=sys.stderr)
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       type=Path, default=Path("outputs/submission.csv"))
    p.add_argument("--candidates",  type=Path, default=Path("data/candidates.jsonl"))
    p.add_argument("--jd",          type=Path, default=Path("data/job_description.md"))
    p.add_argument("--out",         type=Path, default=Path("outputs/submission_reranked.csv"))
    p.add_argument("--model",       default="qwen2.5:7b")
    p.add_argument("--rerank-top",  type=int,  default=200)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading candidates...")
    candidates = {c["candidate_id"]: c for c in load_candidates(args.candidates)}
    jd_text = load_job_description(args.jd)
    jd_snippet = jd_text[:1500]  # keep prompt reasonable

    print(f"Loading ranked CSV from {args.input}...")
    rows = list(csv.DictReader(open(args.input)))
    print(f"  {len(rows)} rows loaded")

    to_rerank = rows[: args.rerank_top]
    keep_as_is = rows[args.rerank_top :]

    print(f"Reranking top {len(to_rerank)} with {args.model}...")
    reranked_scores = {}
    for row in tqdm(to_rerank):
        cid = row["candidate_id"]
        c = candidates.get(cid)
        if not c:
            reranked_scores[cid] = float(row["score"])
            continue
        profile = _build_profile_text(c)
        prompt = RERANK_PROMPT.format(jd=jd_snippet, **profile)
        result = _query_ollama(args.model, prompt)
        if result and "score" in result:
            llm_score = float(result["score"])
            original  = float(row["score"])
            # Blend: 60% LLM, 40% original score — keeps the structural
            # signal from the ranking pipeline and adds LLM's holistic view
            reranked_scores[cid] = 0.6 * llm_score + 0.4 * original
        else:
            reranked_scores[cid] = float(row["score"])

    # Re-sort reranked portion, then append the tail unchanged
    to_rerank.sort(
        key=lambda r: (-reranked_scores[r["candidate_id"]], r["candidate_id"])
    )

    all_rows = to_rerank + keep_as_is
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, row in enumerate(all_rows[:100], start=1):
            cid = row["candidate_id"]
            writer.writerow([
                cid, i,
                f"{reranked_scores.get(cid, float(row['score'])):.6f}",
                row["reasoning"],
            ])

    print(f"\nWrote {args.out}")
    print(f"Validate with: python scripts/validate_submission.py {args.out}")


if __name__ == "__main__":
    main()
