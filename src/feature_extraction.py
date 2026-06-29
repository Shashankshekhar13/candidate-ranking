"""
Feature extraction.

Turns one raw candidate JSON record (per candidate_schema.json) into a flat
dict of derived features that every scoring component reads from. Keeping
this in one place means every component sees the same parsed view of the
candidate and there's a single spot to fix parsing bugs.
"""

from datetime import date, datetime


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _days_since(d, today=None):
    if d is None:
        return None
    today = today or date.today()
    return (today - d).days


def extract_features(candidate: dict, today: date | None = None) -> dict:
    """Flatten a single candidate record into scoring-ready features."""
    today = today or date.today()

    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    # --- career history derived ---
    total_career_months = sum(c.get("duration_months", 0) or 0 for c in career)
    num_employers = len(career)
    current_role = next((c for c in career if c.get("is_current")), career[0] if career else {})
    industries = [c.get("industry", "") or "" for c in career]
    companies = [c.get("company", "") or "" for c in career]
    titles = [c.get("title", "") or "" for c in career]
    career_text = " ".join(
        f"{c.get('title','')} {c.get('company','')} {c.get('industry','')} {c.get('description','')}"
        for c in career
    )

    # overlap detection: sort by start date, check for overlapping ranges
    parsed_ranges = []
    for c in career:
        sd = _parse_date(c.get("start_date"))
        ed = _parse_date(c.get("end_date")) or today
        if sd:
            parsed_ranges.append((sd, ed))
    parsed_ranges.sort()
    has_overlap = any(
        parsed_ranges[i][1] > parsed_ranges[i + 1][0]
        for i in range(len(parsed_ranges) - 1)
    )

    # --- skills derived ---
    skill_names = [s.get("name", "") or "" for s in skills]
    skill_text = " ".join(skill_names)
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    expert_zero_duration = [
        s for s in expert_skills if (s.get("duration_months") or 0) < 3
    ]
    total_endorsements = sum(s.get("endorsements", 0) or 0 for s in skills)

    # --- behavioral signals derived ---
    last_active = _parse_date(signals.get("last_active_date"))
    days_inactive = _days_since(last_active, today)
    signup = _parse_date(signals.get("signup_date"))

    # --- location ---
    location = (profile.get("location") or "").strip().lower()
    country = (profile.get("country") or "").strip().lower()

    full_text = (
        f"{profile.get('headline','')} {profile.get('summary','')} "
        f"{career_text} {skill_text}"
    )

    return {
        "candidate_id": candidate.get("candidate_id"),
        # profile
        "years_of_experience": profile.get("years_of_experience"),
        "current_title": (profile.get("current_title") or "").strip(),
        "current_company": (profile.get("current_company") or "").strip(),
        "current_industry": (profile.get("current_industry") or "").strip(),
        "location": location,
        "country": country,
        # career
        "career": career,
        "career_text": career_text,
        "total_career_months": total_career_months,
        "num_employers": num_employers,
        "industries": industries,
        "companies": companies,
        "titles": titles,
        "current_role_title": (current_role.get("title") or "").strip(),
        "has_career_overlap": has_overlap,
        # education
        "education": education,
        # skills
        "skills": skills,
        "skill_names": skill_names,
        "skill_text": skill_text,
        "num_expert_skills": len(expert_skills),
        "num_expert_zero_duration_skills": len(expert_zero_duration),
        "total_endorsements": total_endorsements,
        # signals
        "signals": signals,
        "days_inactive": days_inactive,
        "signup_date": signup,
        "open_to_work": bool(signals.get("open_to_work_flag")),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "notice_period_days": signals.get("notice_period_days"),
        "willing_to_relocate": bool(signals.get("willing_to_relocate")),
        "github_activity_score": signals.get("github_activity_score"),
        "connection_count": signals.get("connection_count"),
        "endorsements_received": signals.get("endorsements_received"),
        "interview_completion_rate": signals.get("interview_completion_rate"),
        "offer_acceptance_rate": signals.get("offer_acceptance_rate"),
        "profile_completeness_score": signals.get("profile_completeness_score"),
        "search_appearance_30d": signals.get("search_appearance_30d"),
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d"),
        "verified_email": bool(signals.get("verified_email")),
        "verified_phone": bool(signals.get("verified_phone")),
        # combined text for semantic matching
        "full_text": full_text,
    }
