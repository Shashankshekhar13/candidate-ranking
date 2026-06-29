"""
tests/test_pipeline.py

Run with:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_extraction import extract_features
from src.scoring import hard_filters, must_have_skills, behavioral_signal, honeypot_detection, composite


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_candidate(**overrides) -> dict:
    """Minimal valid candidate record; override any field you want to test."""
    base = {
        "candidate_id": "TEST_001",
        "profile": {
            "current_title": "Machine Learning Engineer",
            "current_company": "Acme AI",
            "years_of_experience": 7.0,
            "location": "Pune, Maharashtra",
            "country": "India",
            "headline": "ML Engineer specialising in retrieval and ranking",
            "summary": "5+ years building embedding-based retrieval systems at scale.",
            "current_industry": "AI / ML",
        },
        "career_history": [
            {
                "title": "Senior ML Engineer",
                "company": "Acme AI",
                "industry": "Technology",
                "duration_months": 36,
                "is_current": True,
                "start_date": "2022-01-01",
                "end_date": None,
                "description": (
                    "Built end-to-end vector search pipeline using FAISS and Pinecone. "
                    "Deployed sentence-transformer embeddings for semantic retrieval. "
                    "Evaluated ranking quality using NDCG and MRR metrics."
                ),
            },
            {
                "title": "ML Engineer",
                "company": "StartupX",
                "industry": "E-commerce",
                "duration_months": 48,
                "is_current": False,
                "start_date": "2018-01-01",
                "end_date": "2022-01-01",
                "description": "Recommendation engine using Python and scikit-learn.",
            },
        ],
        "skills": [
            {"name": "Python",      "proficiency": "expert",        "duration_months": 60, "endorsements": 20},
            {"name": "Pinecone",    "proficiency": "advanced",       "duration_months": 24, "endorsements": 8},
            {"name": "FAISS",       "proficiency": "advanced",       "duration_months": 18, "endorsements": 5},
            {"name": "PyTorch",     "proficiency": "intermediate",   "duration_months": 30, "endorsements": 3},
        ],
        "education": [{"degree": "B.Tech", "field_of_study": "Computer Science", "college_tier": 1}],
        "redrob_signals": {
            "last_active_date": "2026-06-15",
            "signup_date": "2020-01-01",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.85,
            "notice_period_days": 30,
            "willing_to_relocate": True,
            "github_activity_score": 72,
            "connection_count": 450,
            "endorsements_received": 36,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.75,
            "profile_completeness_score": 88,
            "search_appearance_30d": 30,
            "saved_by_recruiters_30d": 5,
            "verified_email": True,
            "verified_phone": True,
        },
    }
    # deep-merge overrides
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return base


# ── Feature extraction ────────────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_basic_fields_present(self):
        c = _make_candidate()
        f = extract_features(c)
        assert f["candidate_id"] == "TEST_001"
        assert f["years_of_experience"] == 7.0
        assert "pinecone" in f["skill_text"].lower()
        assert "faiss" in f["career_text"].lower()

    def test_expert_zero_duration_flagged(self):
        c = _make_candidate()
        c["skills"].append({"name": "Weaviate", "proficiency": "expert", "duration_months": 0})
        f = extract_features(c)
        assert f["num_expert_zero_duration_skills"] >= 1

    def test_career_overlap_detected(self):
        c = _make_candidate()
        # Two overlapping full-time roles
        c["career_history"] = [
            {"title": "A", "company": "X", "duration_months": 24,
             "start_date": "2020-01-01", "end_date": "2022-01-01", "is_current": False, "description": ""},
            {"title": "B", "company": "Y", "duration_months": 24,
             "start_date": "2021-01-01", "end_date": "2023-01-01", "is_current": True, "description": ""},
        ]
        f = extract_features(c)
        assert f["has_career_overlap"] is True

    def test_no_career_overlap_on_sequential(self):
        c = _make_candidate()
        # Sequential roles — no overlap
        c["career_history"] = [
            {"title": "A", "company": "X", "duration_months": 24,
             "start_date": "2018-01-01", "end_date": "2020-01-01", "is_current": False, "description": ""},
            {"title": "B", "company": "Y", "duration_months": 24,
             "start_date": "2020-01-01", "end_date": "2022-01-01", "is_current": True, "description": ""},
        ]
        f = extract_features(c)
        assert f["has_career_overlap"] is False


# ── Hard filters ──────────────────────────────────────────────────────────────

class TestHardFilters:
    def test_in_band_experience_scores_high(self):
        f = extract_features(_make_candidate())
        result = hard_filters.score_hard_filters(f)
        assert result["experience_fit"] >= 0.95

    def test_too_junior_experience_scores_low(self):
        c = _make_candidate()
        c["profile"]["years_of_experience"] = 1.0
        f = extract_features(c)
        result = hard_filters.score_hard_filters(f)
        assert result["experience_fit"] < 0.5

    def test_target_city_scores_full(self):
        f = extract_features(_make_candidate())
        result = hard_filters.score_hard_filters(f)
        assert result["location_fit"] == 1.0

    def test_outside_india_scores_low(self):
        c = _make_candidate()
        c["profile"]["country"] = "United States"
        c["profile"]["location"] = "New York"
        c["redrob_signals"]["willing_to_relocate"] = False
        f = extract_features(c)
        result = hard_filters.score_hard_filters(f)
        assert result["location_fit"] < 0.3

    def test_consulting_only_career_penalised(self):
        c = _make_candidate()
        c["career_history"] = [
            {"title": "Senior Consultant", "company": "Infosys", "industry": "IT Services",
             "duration_months": 60, "is_current": True, "start_date": "2019-01-01",
             "end_date": None, "description": "SAP consulting"},
            {"title": "Consultant", "company": "TCS", "industry": "IT Services",
             "duration_months": 36, "is_current": False, "start_date": "2016-01-01",
             "end_date": "2019-01-01", "description": "ERP implementation"},
        ]
        f = extract_features(c)
        result = hard_filters.score_hard_filters(f)
        assert result["disqualifier_multiplier"] < 0.5
        assert any("consulting" in n for n in result["notes"])


# ── Must-have skills ──────────────────────────────────────────────────────────

class TestMustHaveSkills:
    def test_strong_ai_candidate_scores_high(self):
        f = extract_features(_make_candidate())
        result = must_have_skills.score_must_have_skills(f)
        assert result["must_have_score"] >= 0.55
        assert result["embedding_score"] > 0
        assert result["vector_db_score"] > 0

    def test_keyword_stuffer_scores_lower_than_career_match(self):
        """Skills-list-only terms should weigh less than career-history terms."""
        # Candidate A: only skill list mentions, no career history context
        c_a = _make_candidate()
        c_a["career_history"] = [
            {"title": "Marketing Manager", "company": "BrandCo", "industry": "Marketing",
             "duration_months": 72, "is_current": True, "start_date": "2018-01-01",
             "end_date": None, "description": "Managed brand campaigns."}
        ]
        c_a["skills"] = [
            {"name": "Pinecone", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
            {"name": "FAISS",    "proficiency": "expert", "duration_months": 0, "endorsements": 0},
            {"name": "Embedding","proficiency": "expert", "duration_months": 0, "endorsements": 0},
        ]
        f_a = extract_features(c_a)
        score_a = must_have_skills.score_must_have_skills(f_a)["must_have_score"]

        # Candidate B: full career history context
        f_b = extract_features(_make_candidate())
        score_b = must_have_skills.score_must_have_skills(f_b)["must_have_score"]

        assert score_b > score_a, (
            f"Career-contextual candidate ({score_b:.3f}) should outscore "
            f"keyword-stuffer ({score_a:.3f})"
        )


# ── Behavioral signals ────────────────────────────────────────────────────────

class TestBehavioralSignal:
    def test_active_engaged_candidate_near_ceiling(self):
        f = extract_features(_make_candidate())
        result = behavioral_signal.score_behavioral_signal(f)
        assert result["behavioral_multiplier"] >= 0.9

    def test_inactive_candidate_gets_lower_multiplier(self):
        c = _make_candidate()
        c["redrob_signals"]["last_active_date"] = "2025-01-01"
        c["redrob_signals"]["open_to_work_flag"] = False
        c["redrob_signals"]["recruiter_response_rate"] = 0.05
        f = extract_features(c)
        result = behavioral_signal.score_behavioral_signal(f)
        assert result["behavioral_multiplier"] < 0.7


# ── Honeypot detection ────────────────────────────────────────────────────────

class TestHoneypotDetection:
    def test_clean_candidate_not_flagged(self):
        f = extract_features(_make_candidate())
        result = honeypot_detection.score_honeypot(f)
        assert not result["is_flagged"]
        assert result["honeypot_multiplier"] == 1.0

    def test_multiple_expert_zero_duration_skills_flagged(self):
        c = _make_candidate()
        c["skills"] = [
            {"name": s, "proficiency": "expert", "duration_months": 0, "endorsements": 0}
            for s in ["Pinecone","Weaviate","Qdrant","Milvus","FAISS","OpenAI","LangChain","Llama"]
        ]
        f = extract_features(c)
        result = honeypot_detection.score_honeypot(f)
        assert result["is_flagged"]
        assert result["honeypot_multiplier"] < 0.05

    def test_overlapping_career_plus_inflated_endorsements_flagged(self):
        c = _make_candidate()
        c["career_history"] = [
            {"title": "A", "company": "X", "duration_months": 24,
             "start_date": "2020-01-01", "end_date": "2022-01-01", "is_current": False, "description": ""},
            {"title": "B", "company": "Y", "duration_months": 24,
             "start_date": "2021-01-01", "end_date": "2023-01-01", "is_current": True, "description": ""},
        ]
        c["redrob_signals"]["connection_count"] = 10
        c["skills"] = [
            {"name": f"Skill{i}", "proficiency": "intermediate", "duration_months": 12, "endorsements": 40}
            for i in range(5)
        ]
        f = extract_features(c)
        result = honeypot_detection.score_honeypot(f)
        assert result["is_flagged"]


# ── Full composite pipeline ───────────────────────────────────────────────────

class TestComposite:
    def test_strong_candidate_scores_above_07(self):
        f = extract_features(_make_candidate())
        result = composite.score_candidate(f, semantic_fit_score=0.75)
        assert result["final_score"] >= 0.70

    def test_honeypot_candidate_scores_near_zero(self):
        c = _make_candidate()
        c["skills"] = [
            {"name": s, "proficiency": "expert", "duration_months": 0, "endorsements": 0}
            for s in ["Pinecone","Weaviate","Qdrant","Milvus","FAISS","OpenAI","LangChain","Llama"]
        ]
        c["career_history"][0]["start_date"] = "2020-01-01"
        c["career_history"].append({
            "title": "Parallel Role", "company": "Other Co", "duration_months": 24,
            "start_date": "2021-01-01", "end_date": "2023-01-01",
            "is_current": False, "description": ""
        })
        f = extract_features(c)
        result = composite.score_candidate(f, semantic_fit_score=0.9)
        assert result["final_score"] < 0.05

    def test_civil_engineer_scores_lower_than_ml_engineer(self):
        good = _make_candidate()
        bad  = _make_candidate()
        bad["profile"]["current_title"]   = "Civil Engineer"
        bad["profile"]["current_industry"]= "Construction"
        bad["career_history"] = [
            {"title": "Civil Engineer", "company": "BuildCo", "industry": "Construction",
             "duration_months": 84, "is_current": True, "start_date": "2018-01-01",
             "end_date": None, "description": "Structural design and site supervision."}
        ]
        bad["skills"] = [{"name": "AutoCAD", "proficiency": "expert", "duration_months": 48, "endorsements": 5}]

        f_good = extract_features(good)
        f_bad  = extract_features(bad)
        s_good = composite.score_candidate(f_good, 0.75)["final_score"]
        s_bad  = composite.score_candidate(f_bad,  0.20)["final_score"]
        assert s_good > s_bad * 2, f"ML Eng ({s_good:.3f}) should far outscore Civil Eng ({s_bad:.3f})"


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v"])
