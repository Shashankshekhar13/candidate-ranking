import argparse
import csv
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_candidates, load_job_description

POLISH_PROMPT = """\
You are an expert technical recruiter. Your task is to write a polished, professional, and recruiter-friendly candidate evaluation summary (exactly one sentence) for our submission.

We have a draft summary for the candidate:
"{draft_reasoning}"

Here is the candidate's profile:
ID: {candidate_id}
Title: {title}
Company: {company}
Years Experience: {yoe}
Location: {location}
Summary: {summary}
Top Skills: {skills}
Career: {career}

Based on the draft summary and the candidate profile, rewrite the evaluation summary into a single, high-impact sentence. Do NOT mention details not in the profile or draft summary. Keep it strictly to one sentence, concise and highly professional.

IMPORTANT: Respond with ONLY a JSON object in this exact format:
{"reason": "<one sentence polished summary>"}
"""

def _build_profile_text(candidate: dict) -> dict:
    p = candidate.get("profile", {}) or {}
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
        "location": p.get("location", ""),
        "summary": (p.get("summary") or "")[:300],
        "skills":  skills,
        "career":  " | ".join(career_parts),
    }

def _query_ollama(model: str, prompt: str) -> str | None:
    try:
        # pyrefly: ignore [missing-import]
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
        res_json = json.loads(text)
        return res_json.get("reason")
    except Exception as e:
        print(f"  Ollama error: {e}", file=sys.stderr)
        return None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       type=Path, default=Path("outputs/submission.csv"))
    p.add_argument("--candidates",  type=Path, default=Path("data/candidates.jsonl"))
    p.add_argument("--jd",          type=Path, default=Path("data/job_description.md"))
    p.add_argument("--out",         type=Path, default=Path("outputs/submission_ai_polish.csv"))
    p.add_argument("--model",       default="qwen2.5:7b")
    return p.parse_args()

def main():
    args = parse_args()

    print(f"Loading candidates...")
    candidates = {c["candidate_id"]: c for c in load_candidates(args.candidates)}
    
    print(f"Loading submission CSV from {args.input}...")
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    print(f"Polishing reasoning for {len(rows)} candidates...")
    polished_count = 0
    for row in tqdm(rows):
        cid = row["candidate_id"]
        cand = candidates.get(cid)
        if not cand:
            continue
        profile_info = _build_profile_text(cand)
        prompt = POLISH_PROMPT.format(
            draft_reasoning=row["reasoning"],
            candidate_id=cid,
            title=profile_info["title"],
            company=profile_info["company"],
            yoe=profile_info["yoe"],
            location=profile_info["location"],
            summary=profile_info["summary"],
            skills=profile_info["skills"],
            career=profile_info["career"]
        )
        polished_reason = _query_ollama(args.model, prompt)
        if polished_reason:
            row["reasoning"] = polished_reason
            polished_count += 1
            
    print(f"Writing polished output to {args.out}... (Polished {polished_count}/{len(rows)})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

if __name__ == "__main__":
    main()
