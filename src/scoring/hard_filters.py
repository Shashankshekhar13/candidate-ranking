"""
Hard filters / rule layer.

Encodes the disqualifiers and fit bands the JD spells out explicitly. Most
of these are *soft* penalties (multipliers), not hard zeroes -- the JD itself
says several of its rules are "probably not" rather than "never", and a
ranking system should reflect that nuance rather than hard-cutting people.

Returns a dict with:
  - experience_fit (0-1)
  - location_fit (0-1)
  - disqualifier_multiplier (0-1, multiplicative penalty for JD red flags)
  - notes: list of short strings explaining what fired (used by reasoning.py)
"""

from src import config


def _triangular_fit(years):
    """Soft triangular fit around the 5-9 (ideal 6-8) experience band."""
    if years is None:
        return 0.3
    if config.EXPERIENCE_BAND_IDEAL_LOW <= years <= config.EXPERIENCE_BAND_IDEAL_HIGH:
        return 1.0
    if years < config.EXPERIENCE_SOFT_FLOOR or years > config.EXPERIENCE_SOFT_CEILING:
        return 0.15
    if years < config.EXPERIENCE_BAND_IDEAL_LOW:
        span = config.EXPERIENCE_BAND_IDEAL_LOW - config.EXPERIENCE_SOFT_FLOOR
        return 0.15 + 0.85 * (years - config.EXPERIENCE_SOFT_FLOOR) / span
    span = config.EXPERIENCE_SOFT_CEILING - config.EXPERIENCE_BAND_IDEAL_HIGH
    return 1.0 - 0.85 * (years - config.EXPERIENCE_BAND_IDEAL_HIGH) / span


def _location_fit(location: str, country: str, willing_to_relocate: bool):
    if country == config.TARGET_COUNTRY:
        if location in config.TARGET_CITIES or any(c in location for c in config.TARGET_CITIES):
            return 1.0
        # in India but not a named target city
        return 0.75 if willing_to_relocate else 0.55
    # outside India: JD says case-by-case, no visa sponsorship
    return 0.45 if willing_to_relocate else 0.15


def _is_consulting_only(industries, companies):
    if not industries and not companies:
        return False
    text_blobs = [f"{i} {c}".lower() for i, c in zip(industries, companies)]
    return all(
        any(firm in blob for firm in config.CONSULTING_FIRMS) for blob in text_blobs
    )


def _is_pure_research(industries):
    if not industries:
        return False
    return all(
        any(term in (ind or "").lower() for term in config.ACADEMIC_INDUSTRY_TERMS)
        for ind in industries
    )


def _is_senior_title_no_recent_code(current_title, github_score, total_career_months):
    title_l = (current_title or "").lower()
    is_senior_nontechnical_title = any(t in title_l for t in config.SENIOR_TITLE_TERMS)
    # proxy for "hasn't written code in 18 months": senior non-IC title +
    # no observable code activity signal
    no_code_signal = (github_score is None) or (github_score <= 0)
    return is_senior_nontechnical_title and no_code_signal


def _is_cv_speech_robotics_without_nlp(skill_text, career_text):
    text = f"{skill_text} {career_text}".lower()
    has_cv_speech_robotics = any(t in text for t in config.CV_SPEECH_ROBOTICS_TERMS)
    has_nlp_ir = any(t in text for t in config.NLP_IR_TERMS)
    return has_cv_speech_robotics and not has_nlp_ir


def score_hard_filters(features: dict) -> dict:
    notes = []

    experience_fit = _triangular_fit(features.get("years_of_experience"))

    location_fit = _location_fit(
        features.get("location", ""),
        features.get("country", ""),
        features.get("willing_to_relocate", False),
    )

    disqualifier_multiplier = 1.0

    if _is_consulting_only(features.get("industries", []), features.get("companies", [])):
        disqualifier_multiplier *= 0.35
        notes.append("entire career at consulting/IT-services firms only")

    if _is_pure_research(features.get("industries", [])):
        disqualifier_multiplier *= 0.30
        notes.append("pure research/academic background, no production deployment")

    if _is_senior_title_no_recent_code(
        features.get("current_title", ""),
        features.get("github_activity_score"),
        features.get("total_career_months", 0),
    ):
        disqualifier_multiplier *= 0.55
        notes.append("senior/architect title with no observable recent coding signal")

    if _is_cv_speech_robotics_without_nlp(
        features.get("skill_text", ""), features.get("career_text", "")
    ):
        disqualifier_multiplier *= 0.45
        notes.append("CV/speech/robotics background without NLP/IR exposure")

    # notice period: JD wants sub-30-day, can buy out up to 30; 30+ still in
    # scope but "bar gets higher" -- soft penalty, not elimination
    notice = features.get("notice_period_days")
    if notice is not None and notice > 60:
        disqualifier_multiplier *= 0.85
        notes.append(f"long notice period ({notice} days)")

    return {
        "experience_fit": experience_fit,
        "location_fit": location_fit,
        "disqualifier_multiplier": disqualifier_multiplier,
        "notes": notes,
    }
