"""
Composite scorer.

final_score = (
    base_fit (weighted sum of must-have-skills, semantic-fit, experience-fit, location-fit)
    * disqualifier_multiplier      (hard_filters.py — JD red flags)
    * behavioral_multiplier        (behavioral_signal.py — engagement/availability)
    * honeypot_multiplier          (honeypot_detection.py — anomaly penalty)
)

Multiplicative combination (rather than purely additive) is deliberate: a
candidate who is a perfect skills match but is a flagged honeypot, or who is
a perfect skills match but hasn't logged in for a year, should not simply
have a few points subtracted -- their effective rank should collapse. An
additive model lets a high base score paper over a serious red flag;
multiplying does not.
"""

from src import config
from src.scoring import behavioral_signal, hard_filters, honeypot_detection, must_have_skills


def score_candidate(features: dict, semantic_fit_score: float) -> dict:
    hf = hard_filters.score_hard_filters(features)
    mh = must_have_skills.score_must_have_skills(features)
    bh = behavioral_signal.score_behavioral_signal(features)
    hp = honeypot_detection.score_honeypot(features)

    # Core skill fit is the gate: how relevant is this person to the role,
    # independent of where they live or exactly how many years they have.
    core_fit = (
        config.WEIGHT_MUST_HAVE_SKILLS * mh["must_have_score"]
        + config.WEIGHT_SEMANTIC_FIT * semantic_fit_score
    )

    # Experience and location modulate core_fit multiplicatively -- a
    # candidate with zero skill signal can't be rescued by being in Pune
    # with exactly 7 years of experience, but a strong skill match still
    # gets meaningfully boosted by being well-located and in-band.
    exp_floor = config.EXPERIENCE_MULTIPLIER_FLOOR
    loc_floor = config.LOCATION_MULTIPLIER_FLOOR
    experience_multiplier = exp_floor + (1 - exp_floor) * hf["experience_fit"]
    location_multiplier = loc_floor + (1 - loc_floor) * hf["location_fit"]

    base_fit = core_fit * experience_multiplier * location_multiplier

    final_score = (
        base_fit
        * hf["disqualifier_multiplier"]
        * bh["behavioral_multiplier"]
        * hp["honeypot_multiplier"]
    )

    notes = list(hf["notes"]) + list(mh["notes"]) + list(bh["notes"])
    if hp["is_flagged"]:
        notes.append(f"HONEYPOT FLAGGED: {', '.join(hp['flags_fired'])}")

    return {
        "candidate_id": features["candidate_id"],
        "final_score": final_score,
        "base_fit": base_fit,
        "core_fit": core_fit,
        "semantic_fit_score": semantic_fit_score,
        "must_have_score": mh["must_have_score"],
        "experience_fit": hf["experience_fit"],
        "location_fit": hf["location_fit"],
        "disqualifier_multiplier": hf["disqualifier_multiplier"],
        "behavioral_multiplier": bh["behavioral_multiplier"],
        "honeypot_multiplier": hp["honeypot_multiplier"],
        "is_honeypot_flagged": hp["is_flagged"],
        "notes": notes,
        "component_breakdown": {
            "embedding_score": mh["embedding_score"],
            "vector_db_score": mh["vector_db_score"],
            "eval_score": mh["eval_score"],
            "python_score": mh["python_score"],
        },
    }
