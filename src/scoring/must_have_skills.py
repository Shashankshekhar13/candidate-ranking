"""
Must-have skills component.

The JD's "things you absolutely need" section: production embeddings-based
retrieval, vector DB / hybrid search infra, strong Python, and evaluation
framework experience for ranking systems.

Key design choice: a term appearing in `career_text` (actual job
descriptions) counts far more than the same term appearing only in the
`skills` list. This is the direct countermeasure to the keyword-stuffer
trap the JD calls out -- "all the AI keywords listed as skills but whose
title is Marketing Manager is not a fit". A skill list mention is cheap to
fake; a specific sentence in a job description about deploying it is not.
"""

from src import config


def _term_hits(text: str, terms: list[str]) -> int:
    text = (text or "").lower()
    return sum(1 for t in terms if t in text)


def _skill_context_score(features: dict, terms: list[str]) -> float:
    """
    Score in [0, 1] for how strongly a term family shows up, weighted
    3x for career-history (real, contextual) mentions vs skill-list-only
    (cheap, list) mentions.
    """
    career_hits = _term_hits(features.get("career_text", ""), terms)
    skill_hits = _term_hits(features.get("skill_text", ""), terms)

    # also check skills[] entries for actual depth (duration + endorsements)
    # on the matching skill names, not just presence
    depth_bonus = 0.0
    for s in features.get("skills", []):
        name = (s.get("name") or "").lower()
        if any(t in name for t in terms):
            duration = s.get("duration_months", 0) or 0
            endorsements = s.get("endorsements", 0) or 0
            if duration >= 12:
                depth_bonus += 0.15
            if endorsements >= 5:
                depth_bonus += 0.10

    raw = 3.0 * min(career_hits, 3) + 1.0 * min(skill_hits, 3) + depth_bonus
    # squash to 0-1 with a soft cap; 3 career hits + decent depth ~= 1.0
    return min(raw / 10.0, 1.0)


def score_must_have_skills(features: dict) -> dict:
    embedding_score = _skill_context_score(features, config.EMBEDDING_RETRIEVAL_TERMS)
    vector_db_score = _skill_context_score(features, config.VECTOR_DB_TERMS)
    eval_score = _skill_context_score(features, config.EVAL_FRAMEWORK_TERMS)

    python_in_skills = any(
        "python" in (s.get("name") or "").lower() for s in features.get("skills", [])
    )
    python_in_career = "python" in (features.get("career_text", "") or "").lower()
    python_score = 1.0 if (python_in_skills or python_in_career) else 0.2

    # weighted: embeddings + vector DB are the two non-negotiables, eval
    # framework experience and python round it out
    must_have_score = (
        0.35 * embedding_score
        + 0.30 * vector_db_score
        + 0.20 * eval_score
        + 0.15 * python_score
    )

    notes = []
    if embedding_score < 0.2:
        notes.append("little/no production embeddings-retrieval evidence")
    elif embedding_score >= 0.6:
        notes.append("strong embeddings/retrieval experience in career history")
    if vector_db_score < 0.2:
        notes.append("little/no vector-DB or hybrid-search evidence")
    if eval_score >= 0.4:
        notes.append("has ranking-evaluation framework experience")

    return {
        "must_have_score": must_have_score,
        "embedding_score": embedding_score,
        "vector_db_score": vector_db_score,
        "eval_score": eval_score,
        "python_score": python_score,
        "notes": notes,
    }
