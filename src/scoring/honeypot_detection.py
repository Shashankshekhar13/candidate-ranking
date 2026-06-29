"""
Honeypot / anomaly detection.

The dataset contains ~80 honeypot candidates with subtly impossible
profiles. Ranking any of these into the top 100 risks the >10% honeypot-rate
disqualification at Stage 3. This module runs a handful of independent,
explainable consistency checks -- not a black box -- so each flag can be
quoted directly in the reasoning column.

Each check is intentionally conservative (low false-positive rate on
plausible-but-unusual real candidates) and independent of the others, so we
require config.HONEYPOT_FLAG_THRESHOLD checks to fire before applying the
penalty. A single odd field is normal noise in 100k records; multiple
independent red flags on the same candidate is the honeypot signature.
"""

from src import config


def _check_expert_zero_duration(features):
    """'Expert' proficiency with ~0 months of experience using the skill."""
    return features.get("num_expert_zero_duration_skills", 0) >= 1


def _check_too_many_experts(features):
    """Implausibly broad expertise: many 'expert'-rated skills at once."""
    return features.get("num_expert_skills", 0) >= 8


def _check_career_overlap(features):
    """Two or more full-time roles with overlapping date ranges."""
    return bool(features.get("has_career_overlap"))


def _check_experience_vs_career_months(features):
    """
    years_of_experience should roughly match the sum of career_history
    durations (allowing slack for gaps/rounding). A large mismatch in
    either direction is suspicious.
    """
    yoe = features.get("years_of_experience")
    months = features.get("total_career_months", 0)
    if yoe is None or months == 0:
        return False
    implied_years = months / 12.0
    # allow 40% slack for employment gaps, rounding, concurrent part-time etc.
    return implied_years < yoe * 0.5 or implied_years > yoe * 1.6


def _check_endorsements_vs_connections(features):
    """Endorsements wildly exceeding the candidate's connection count."""
    endorsements = features.get("total_endorsements", 0)
    connections = features.get("connection_count") or 0
    if connections == 0:
        return endorsements >= 20
    return endorsements > connections * 4


def _check_completeness_vs_content(features):
    """A near-100% 'complete' profile that's actually thin on substance."""
    completeness = features.get("profile_completeness_score")
    if completeness is None or completeness < 95:
        return False
    thin = (
        len(features.get("skills", [])) <= 1
        or len(features.get("career", [])) == 0
        or len(features.get("education", [])) == 0
    )
    return thin


def _check_tenure_vs_company_size_mismatch(features):
    """
    A very short tenure at the *current* role combined with a title/seniority
    jump that the rest of the career history doesn't support (crude proxy:
    current role < 3 months but candidate claims 8+ years total experience
    AND current title contains a senior marker not present anywhere earlier).
    """
    career = features.get("career", [])
    if not career:
        return False
    current = next((c for c in career if c.get("is_current")), None)
    if not current:
        return False
    duration = current.get("duration_months", 0) or 0
    yoe = features.get("years_of_experience") or 0
    title = (current.get("title") or "").lower()
    senior_markers = ("senior", "lead", "principal", "staff", "head", "director")
    is_senior_title = any(m in title for m in senior_markers)
    prior_titles = " ".join((c.get("title") or "").lower() for c in career if c is not current)
    had_senior_before = any(m in prior_titles for m in senior_markers)
    return duration < 3 and yoe >= 8 and is_senior_title and not had_senior_before


_CHECKS = [
    ("expert_skill_zero_duration", _check_expert_zero_duration),
    ("implausibly_many_expert_skills", _check_too_many_experts),
    ("overlapping_employment_dates", _check_career_overlap),
    ("experience_years_vs_career_history_mismatch", _check_experience_vs_career_months),
    ("endorsements_far_exceed_connections", _check_endorsements_vs_connections),
    ("profile_complete_but_substantively_empty", _check_completeness_vs_content),
    ("unsupported_seniority_jump_in_current_role", _check_tenure_vs_company_size_mismatch),
]


def score_honeypot(features: dict) -> dict:
    fired = [name for name, fn in _CHECKS if fn(features)]
    is_honeypot = len(fired) >= config.HONEYPOT_FLAG_THRESHOLD
    multiplier = config.HONEYPOT_SCORE_MULTIPLIER if is_honeypot else 1.0
    return {
        "honeypot_multiplier": multiplier,
        "is_flagged": is_honeypot,
        "flags_fired": fired,
    }
