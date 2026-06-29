"""
Behavioral signal multiplier.

Per the JD: "a perfect-on-paper candidate who hasn't logged in for 6 months
and has a 5% recruiter response rate is, for hiring purposes, not actually
available." This component turns that into a multiplier on the base fit
score (not an additive term), because the JD frames it explicitly as a
*modifier* on skill-match, not a competing signal.

Output is bounded to [BEHAVIORAL_MULTIPLIER_FLOOR, BEHAVIORAL_MULTIPLIER_CEILING]
so behavioral signals alone can never zero out an otherwise excellent match,
but can meaningfully demote a disengaged candidate.
"""

from src import config


def _clip01(x):
    if x is None:
        return 0.5
    return max(0.0, min(1.0, x))


def _recency_score(days_inactive):
    """1.0 if active recently, decaying to ~0 by 180+ days inactive."""
    if days_inactive is None:
        return 0.5
    if days_inactive <= 7:
        return 1.0
    if days_inactive >= 180:
        return 0.0
    return 1.0 - (days_inactive - 7) / (180 - 7)


def score_behavioral_signal(features: dict) -> dict:
    recency = _recency_score(features.get("days_inactive"))
    response_rate = _clip01(features.get("recruiter_response_rate"))
    open_to_work = 1.0 if features.get("open_to_work") else 0.4
    interview_completion = _clip01(features.get("interview_completion_rate"))

    notice = features.get("notice_period_days")
    if notice is None:
        notice_score = 0.5
    else:
        # JD: sub-30 day is ideal, can buy out up to 30, 30+ still in scope
        notice_score = 1.0 if notice <= 30 else max(0.3, 1.0 - (notice - 30) / 150)

    search_appearance = features.get("search_appearance_30d") or 0
    saved_by_recruiters = features.get("saved_by_recruiters_30d") or 0
    demand_signal = min(1.0, (search_appearance / 50.0) * 0.5 + (saved_by_recruiters / 10.0) * 0.5)

    verification = (
        (1 if features.get("verified_email") else 0)
        + (1 if features.get("verified_phone") else 0)
    ) / 2.0

    engagement_score = (
        0.30 * recency
        + 0.25 * response_rate
        + 0.15 * open_to_work
        + 0.10 * interview_completion
        + 0.10 * notice_score
        + 0.05 * demand_signal
        + 0.05 * verification
    )

    # map engagement_score [0,1] to multiplier range
    lo, hi = config.BEHAVIORAL_MULTIPLIER_FLOOR, config.BEHAVIORAL_MULTIPLIER_CEILING
    multiplier = lo + engagement_score * (hi - lo)

    notes = []
    days_inactive = features.get("days_inactive")
    if days_inactive is not None and days_inactive > 120:
        notes.append(f"inactive for {days_inactive} days")
    if features.get("recruiter_response_rate") is not None and features["recruiter_response_rate"] < 0.2:
        notes.append(f"low recruiter response rate ({features['recruiter_response_rate']:.0%})")
    if not features.get("open_to_work"):
        notes.append("not flagged open to work")
    if notice is not None and notice <= 30:
        notes.append(f"short notice period ({notice} days) — available quickly")

    return {
        "behavioral_multiplier": multiplier,
        "engagement_score": engagement_score,
        "notes": notes,
    }
