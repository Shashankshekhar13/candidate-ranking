"""
Central config for the Redrob ranker.

Every "magic number" used by the scoring components lives here so the
methodology is auditable in one place (useful for the Stage 5 interview —
you should be able to point at this file and explain every choice).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

CANDIDATES_PATH = DATA_DIR / "candidates.jsonl"
JD_PATH = DATA_DIR / "job_description.md"

TOP_K = 100

# ---------------------------------------------------------------------------
# Composite score weights
# ---------------------------------------------------------------------------
# Core skill fit (must sum to 1.0) -- this is the GATE. A candidate with no
# real AI/ML signal should not be rescuable by good location/experience,
# because the JD is explicit that skills-relevance is the primary axis and
# location/experience are secondary modifiers ("Location: ... flexible",
# "5-9 years ... a range, not a requirement"). Location and experience are
# therefore applied as multipliers ON TOP of core skill fit below, not as
# additive terms competing with it -- see scoring/composite.py.
WEIGHT_MUST_HAVE_SKILLS = 0.55   # embeddings/vector-db/python/eval production exp.
WEIGHT_SEMANTIC_FIT = 0.45       # TF-IDF/LSA similarity to JD text

assert abs(WEIGHT_MUST_HAVE_SKILLS + WEIGHT_SEMANTIC_FIT - 1.0) < 1e-9

# Multiplier floors: even a 0-fit candidate on these axes isn't zeroed out
# (JD explicitly treats both as soft/flexible), but a strong candidate on
# both axes gets the full 1.0x.
EXPERIENCE_MULTIPLIER_FLOOR = 0.5   # "a range, not a requirement"
LOCATION_MULTIPLIER_FLOOR = 0.55    # "flexible", case-by-case outside India

# Behavioral multiplier range: final_score = base_fit * behavioral_multiplier
# Keeping the floor above 0 means behavior alone can never zero out an
# otherwise-excellent candidate, but a fully disengaged candidate (floor)
# gets cut to ~40% of their base fit -- usually enough to push them out of
# the top 100 if anything else is competitive.
BEHAVIORAL_MULTIPLIER_FLOOR = 0.40
BEHAVIORAL_MULTIPLIER_CEILING = 1.10

# Honeypot / anomaly penalty: multiplicative, applied after everything else.
HONEYPOT_FLAG_THRESHOLD = 2          # number of independent anomaly checks that must fire
HONEYPOT_SCORE_MULTIPLIER = 0.01     # near-zero, not literally zero (keeps sort stable)

# ---------------------------------------------------------------------------
# JD-derived constants
# ---------------------------------------------------------------------------

# "5-9 years" with an ideal center around 6-8 (per the "how to read between
# the lines" section). Soft triangular fit, not a hard cutoff -- JD explicitly
# says outside-band candidates are still considered if other signals are strong.
EXPERIENCE_BAND_MIN = 5
EXPERIENCE_BAND_IDEAL_LOW = 6
EXPERIENCE_BAND_IDEAL_HIGH = 8
EXPERIENCE_BAND_MAX = 9
EXPERIENCE_SOFT_FLOOR = 2     # below this, fit score bottoms out hard
EXPERIENCE_SOFT_CEILING = 15  # above this, fit score bottoms out hard

# Target cities called out explicitly as welcome to apply
TARGET_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr",
    "gurgaon", "gurugram", "new delhi",
}
TARGET_COUNTRY = "india"

# Consulting-only-career penalty list
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
}

# Must-have production skill families (regex-ish substrings, lowercase)
EMBEDDING_RETRIEVAL_TERMS = [
    "sentence-transformer", "sentence transformer", "openai embedding",
    "bge", "e5 embedding", "embedding", "dense retrieval", "semantic search",
]
VECTOR_DB_TERMS = [
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "vector db", "hybrid search",
]
EVAL_FRAMEWORK_TERMS = [
    "ndcg", "mrr", "map", "a/b test", "ab test", "offline eval",
    "online eval", "evaluation framework", "learning to rank", "learning-to-rank",
]
PYTHON_TERMS = ["python"]

# Things the JD explicitly does NOT want
CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "speech recognition", "robotics", "autonomous driving",
    "image classification", "object detection", "slam",
]
NLP_IR_TERMS = [
    "nlp", "natural language", "retrieval", "ranking", "search", "llm",
    "information retrieval", "text classification", "rag",
]
SENIOR_TITLE_TERMS = ["architect", "director", "vp ", "vice president", "head of"]

ACADEMIC_INDUSTRY_TERMS = ["research", "academia", "university", "academic"]
PRODUCT_COMPANY_NEGATIVE_TERMS = ACADEMIC_INDUSTRY_TERMS + ["it services", "consulting"]
