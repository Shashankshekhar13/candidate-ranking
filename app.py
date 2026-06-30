import csv
import json
import sys
import time
from io import StringIO
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st

st.set_page_config(
    page_title="TalentLens AI",
    page_icon=" ",
    layout="wide",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.data_loader import load_candidates, load_job_description, iter_candidates
from src.feature_extraction import extract_features
from src.reasoning import generate_reasoning
from src.scoring import composite
from src.scoring.semantic_fit import fit_semantic_space, score_semantic_fit_batch
import requests

# CHANGE this to match your repo from Hugging Face
HF_REPO_ID = "Realshashank/talentlens-data"

@st.cache_resource(show_spinner=False)
def ensure_data_files():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    files = {
        "candidates.jsonl": f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/candidates.jsonl",
        "embeddings_cache.npz": f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/embeddings_cache.npz",
    }

    for filename, url in files.items():
        local_path = data_dir / filename
        
        # Check expected size from HEAD request
        try:
            r_head = requests.head(url, timeout=15, allow_redirects=True)
            r_head.raise_for_status()
            expected_size = int(r_head.headers.get("content-length", 0))
        except Exception:
            expected_size = 0

        # If file exists and size matches, skip download
        if local_path.exists():
            if expected_size == 0 or local_path.stat().st_size == expected_size:
                continue
            else:
                # Delete partial/corrupted file
                try:
                    local_path.unlink()
                except Exception:
                    pass

        progress_text = st.empty()
        progress_text.info(f"First-time setup: downloading {filename}...")

        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0

            bar = st.progress(0)
            last_pct = 0
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int((downloaded / total) * 100)
                        if pct >= last_pct + 5:
                            bar.progress(pct / 100.0)
                            last_pct = pct

            bar.empty()
            progress_text.empty()
        except Exception as e:
            # Clean up partial file on failure to avoid leaving a corrupted file
            if local_path.exists():
                try:
                    local_path.unlink()
                except Exception:
                    pass
            progress_text.error(f"Failed to download {filename}: {e}")
            st.stop()

    return True

DOMAIN_MAP = {
    "zomato": "zomato.com",
    "sarvam ai": "sarvam.ai",
    "sarvam": "sarvam.ai",
    "swiggy": "swiggy.in",
    "wysa": "wysa.io",
    "meesho": "meesho.com",
    "razorpay": "razorpay.com",
    "flipkart": "flipkart.com",
    "paytm": "paytm.com",
    "phonepe": "phonepe.com",
    "freshworks": "freshworks.com",
    "genpact": "genpact.com",
    "amazon": "amazon.com",
    "google": "google.com",
    "microsoft": "microsoft.com",
    "openai": "openai.com",
    "nvidia": "nvidia.com",
    "infosys": "infosys.com",
    "wipro": "wipro.com",
    "tcs": "tcs.com",
    "cognizant": "cognizant.com",
    "accenture": "accenture.com",
    "ola": "olacabs.com",
    "byju": "byjus.com",
    "nykaa": "nykaa.com",
    "zepto": "zeptonow.com",
    "cred": "cred.club",
    "groww": "groww.in",
    "mphasis": "mphasis.com",
    "persistent": "persistent.com",
    "hexaware": "hexaware.com",
    "mindtree": "mindtree.com",
    "zensar": "zensar.com",
}

COLORS = [
    "#6366f1","#3b82f6","#10b981","#f59e0b",
    "#ef4444","#8b5cf6","#06b6d4","#ec4899",
]

def company_logo_html(company: str, size: int = 36) -> str:
    key = company.lower().strip()

    # Find domain from map
    domain = None
    for k, v in DOMAIN_MAP.items():
        if k in key:
            domain = v
            break

    # Fallback domain guess
    if not domain:
        clean = (key.replace("private limited","").replace("pvt ltd","")
                    .replace("pvt.ltd","").replace("ltd","")
                    .replace("inc","").strip())
        domain = clean.replace(" ", "") + ".com"

    # Google favicon service — works for any domain, no JS needed
    logo_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"

    # Initials badge as visual backup (shown side by side, Google logo overlays it)
    initials = "".join(w[0].upper() for w in company.split()[:2])
    color = COLORS[sum(ord(c) for c in key) % len(COLORS)]

    return (
        f'<span style="position:relative;display:inline-flex;align-items:center;'
        f'justify-content:center;width:36px;height:36px;border-radius:8px;'
        f'background:{color};color:#fff;font-weight:700;font-size:12px;'
        f'margin-right:10px;vertical-align:middle;flex-shrink:0;overflow:hidden;">'
        f'<span style="z-index:1;">{initials}</span>'
        f'<img src="{logo_url}" width="32" height="32" '
        f'style="position:absolute;top:2px;left:2px;width:32px;height:32px;'
        f'object-fit:contain;border-radius:6px;z-index:2;" />'
        f'</span>'
    )


# ── Custom CSS for Premium Design ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Overrides */
    .stApp {
        background-color: #0b0f19;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
    }
    
    /* Metrics / KPI container */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 28px;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #2d3748;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #6366f1, #3b82f6);
    }
    
    .kpi-val {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 6px;
        font-family: 'Outfit', sans-serif;
    }
    
    .kpi-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }
    
    .kpi-subtext {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Candidate list styling */
    .candidate-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.25s ease-in-out;
    }
    
    .candidate-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.4);
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: start;
        margin-bottom: 12px;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    .card-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .badge-score {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 6px 16px;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .badge-score-high {
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
    }
    
    .badge-score-mid {
        background: rgba(245, 158, 11, 0.1);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.25);
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.15);
    }
    
    .badge-score-low {
        background: rgba(239, 68, 68, 0.1);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.25);
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.15);
    }
    
    .badge-meta {
        background: #1f2937;
        color: #d1d5db;
        border: 1px solid #374151;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    
    .meta-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 16px;
    }

    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 14px;
    }
    
    .tag-skill {
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    
    .tag-skill-matched {
        background: rgba(99, 102, 241, 0.15);
        color: #c7d2fe;
        border: 1px solid rgba(99, 102, 241, 0.35);
    }
    
    .tag-skill-unmatched {
        background: #1f2937;
        color: #9ca3af;
        border: 1px solid #374151;
    }
    
    .ai-reasoning {
        background: #0f172a;
        border-left: 4px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.55;
        font-style: italic;
        margin-top: 14px;
    }
    
    .warning-banner {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #fca5a5;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Sub-score grid */
    .score-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        background: #1f2937;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 16px;
        border: 1px solid #374151;
    }
    
    .score-grid-item {
        display: flex;
        flex-direction: column;
    }
    
    .score-grid-label {
        font-size: 0.72rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 3px;
    }
    
    .score-grid-val {
        font-size: 1rem;
        font-weight: 700;
        color: #f3f4f6;
        font-family: 'Outfit', sans-serif;
    }

    /* Landing / Overview cards */
    .landing-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 26px;
        height: 100%;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .landing-icon {
        font-size: 2.2rem;
        margin-bottom: 18px;
        background: linear-gradient(135deg, #6366f1, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Scrollbar tweak */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0f19;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("TalentLens AI")
st.caption("An AI recruiter that understands candidates, not just keywords.")

# ── File Paths & Preload checks ───────────────────────────────────────────────
cache_path = config.DATA_DIR / "embeddings_cache.npz"
default_jd_text = load_job_description(config.JD_PATH)

# ── Sidebar — Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Sourcing")
    
    use_defaults = st.checkbox(
        "Use dataset files from data/ folder",
        value=True,
        help="If checked, ignores uploads and uses data/candidates.jsonl and data/job_description.md",
    )
    
    cand_file = None
    jd_file = None
    if not use_defaults:
        cand_file = st.file_uploader(
            "Upload candidates.jsonl",
            type=["jsonl", "json"],
            help="The candidate pool from the dataset.",
        )
        jd_file = st.file_uploader(
            "Upload job_description (.md or .txt)",
            type=["md", "txt", "docx"],
            help="The job description to rank against.",
        )

    st.markdown("---")
    st.subheader("🔍 Initial Engine Mode")
    
    default_engine = st.session_state.get("initial_engine_mode", "BGE Dense Vectors (Precomputed)")
    engine_options = ["BGE Dense Vectors (Precomputed)", "TF-IDF + LSA (Bag of Words)"]
    try:
        default_idx = engine_options.index(default_engine)
    except ValueError:
        default_idx = 0

    if cache_path.exists() or use_defaults:
        initial_engine = st.selectbox(
            "Semantic Engine",
            engine_options,
            index=default_idx,
            help="BGE dense embeddings represent semantic intent. TF-IDF + LSA models keyword matching patterns."
        )
    else:
        st.selectbox("Semantic Engine", ["TF-IDF + LSA (Bag of Words)"], disabled=True)
        st.caption("Precomputed BGE cache not found. Using local TF-IDF engine.")
        initial_engine = "TF-IDF + LSA (Bag of Words)"

    st.markdown("---")
    run_btn = st.button("▶ Run Scoring Pipeline", type="primary", use_container_width=True)
    
    if "scored_candidates" in st.session_state:
        st.success("✅ Active workspace loaded")
        st.caption(f"Engine: **{st.session_state.model_used}**")
        st.caption(f"Loaded size: **{len(st.session_state.features_list):,}** candidates")

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_from_disk():
    candidates = load_candidates(config.CANDIDATES_PATH)
    jd_text    = load_job_description(config.JD_PATH)
    return candidates, jd_text

def _load_from_upload(cand_bytes, jd_bytes):
    text = cand_bytes.decode("utf-8")
    candidates = [json.loads(l) for l in text.splitlines() if l.strip()]
    jd_text = jd_bytes.decode("utf-8")
    return candidates, jd_text

# ── Main Controller ───────────────────────────────────────────────────────────
if run_btn:
    st.session_state.start_time = time.time()
    
    # 1. Load JD Text
    try:
        if use_defaults:
            ensure_data_files()
            jd_text = load_job_description(config.JD_PATH)
        else:
            if not cand_file or not jd_file:
                st.error("Please upload both candidate pool and JD files.")
                st.stop()
            jd_text = jd_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Failed to load JD text: {e}")
        st.stop()
        
    st.session_state.jd_text = jd_text
    
    # 2. Determine semantic model & scores
    try:
        if use_defaults:
            use_bge = "BGE" in initial_engine
            jd_changed = jd_text.strip() != default_jd_text.strip()
            
            if use_bge and cache_path.exists() and not jd_changed:
                with st.spinner("Loading dense embeddings cache..."):
                    data = np.load(cache_path, allow_pickle=True)
                    c_emb = data["candidate_embeddings"]
                    jd_emb = data["jd_embedding"]
                    sims = (c_emb @ jd_emb.T).squeeze()
                    semantic_scores = (sims + 1.0) / 2.0
                    st.session_state.cache_used = True
                    st.session_state.model_used = "BGE Dense Semantic Cache"
            else:
                st.warning("Running TF-IDF on 100k candidates may exceed the 1GB RAM limit. If the app crashes, please use the default BGE engine.")
                with st.spinner("Loading candidate profiles for TF-IDF..."):
                    candidates = load_candidates(config.CANDIDATES_PATH)
                with st.spinner("Fitting TF-IDF / LSA space..."):
                    texts = []
                    for c in candidates:
                        feats = extract_features(c)
                        texts.append(feats["full_text"])
                    _, _, jd_vec, c_vecs = fit_semantic_space(jd_text, texts)
                    semantic_scores = score_semantic_fit_batch(jd_vec, c_vecs)
                    st.session_state.cache_used = False
                    st.session_state.model_used = "TF-IDF + LSA (Fallback)"
        else:
            # Uploaded files are small, so load them fully
            st.session_state.cache_used = False
            st.session_state.model_used = "TF-IDF + LSA (Uploaded Data)"
            cand_text = cand_file.read().decode("utf-8")
            candidates = [json.loads(l) for l in cand_text.splitlines() if l.strip()]
            texts = []
            for c in candidates:
                feats = extract_features(c)
                texts.append(feats["full_text"])
            _, _, jd_vec, c_vecs = fit_semantic_space(jd_text, texts)
            semantic_scores = score_semantic_fit_batch(jd_vec, c_vecs)
    except Exception as e:
        st.error(f"Failed to calculate semantic space: {e}")
        st.stop()

    # 3. Stream, score, and aggregates
    with st.spinner("Streaming, scoring, and verifying candidate pool..."):
        total_candidates = 0
        yoe_segment_counts = {"Entry (<2y)": 0, "Mid-level (2-5y)": 0, "Ideal Band (5-9y)": 0, "Senior (9-12y)": 0, "Executive (12y+)": 0}
        loc_segment_counts = {"Target Cities (India)": 0, "Other India": 0, "Global": 0}
        hp_flags_fired = []
        total_hp = 0
        top_candidates = []
        
        # Generator for streaming
        if use_defaults:
            cand_stream = iter_candidates(config.CANDIDATES_PATH)
        else:
            cand_stream = candidates
            
        for idx_row, c in enumerate(cand_stream):
            total_candidates += 1
            feats = extract_features(c)
            sem_score = semantic_scores[idx_row]
            
            res = composite.score_candidate(feats, float(sem_score))
            res["features"] = feats
            
            # Running aggregates
            yoe = feats.get("years_of_experience")
            if yoe is not None:
                if yoe < 2:
                    yoe_segment_counts["Entry (<2y)"] += 1
                elif yoe <= 5:
                    yoe_segment_counts["Mid-level (2-5y)"] += 1
                elif yoe <= 9:
                    yoe_segment_counts["Ideal Band (5-9y)"] += 1
                elif yoe <= 12:
                    yoe_segment_counts["Senior (9-12y)"] += 1
                else:
                    yoe_segment_counts["Executive (12y+)"] += 1
                    
            loc = (feats.get("location") or "").lower()
            country = (feats.get("country") or "").lower()
            in_target = any(city in loc for city in config.TARGET_CITIES)
            in_india = country == "india" or any(city in loc for city in config.TARGET_CITIES)
            if in_target:
                loc_segment_counts["Target Cities (India)"] += 1
            elif in_india:
                loc_segment_counts["Other India"] += 1
            else:
                loc_segment_counts["Global"] += 1
                
            if res["is_honeypot_flagged"]:
                total_hp += 1
                for note in res["notes"]:
                    if "HONEYPOT FLAGGED" in note:
                        flags = note.replace("HONEYPOT FLAGGED:", "").strip().split(", ")
                        hp_flags_fired.extend(flags)
            
            top_candidates.append(res)
            if len(top_candidates) > 300:
                top_candidates.sort(key=lambda r: (-round(r["final_score"], 6), r["candidate_id"]))
                top_candidates = top_candidates[:200]
                
        # Final sort and store
        top_candidates.sort(key=lambda r: (-round(r["final_score"], 6), r["candidate_id"]))
        
        st.session_state.scored_candidates = top_candidates
        st.session_state.total_candidates = total_candidates
        st.session_state.yoe_counts = pd.Series(yoe_segment_counts)
        st.session_state.loc_counts = pd.Series(loc_segment_counts)
        st.session_state.total_hp = total_hp
        st.session_state.hp_counts = pd.Series(hp_flags_fired).value_counts() if hp_flags_fired else pd.Series()
        
        st.session_state.elapsed_time = time.time() - st.session_state.start_time
        st.session_state.has_run = True
        
    st.balloons()
    st.rerun()

# ── Render UI ─────────────────────────────────────────────────────────────────
if "has_run" in st.session_state and st.session_state.has_run:
    scored_candidates = st.session_state.scored_candidates
    elapsed = st.session_state.elapsed_time
    total_hp = st.session_state.total_hp
    top_score = scored_candidates[0]["final_score"] if scored_candidates else 0
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-val">{st.session_state.total_candidates:,}</div>
            <div class="kpi-label">Talent Pool Scored</div>
            <div class="kpi-subtext">All candidate profiles parsed</div>
            <div class="kpi-icon">👥</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">{top_score:.1%}</div>
            <div class="kpi-label">Highest Score Match</div>
            <div class="kpi-subtext">Strongest recruiter alignment</div>
            <div class="kpi-icon">🏆</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">{elapsed:.2f}s</div>
            <div class="kpi-label">Compute Wall-clock</div>
            <div class="kpi-subtext">Budget limits: &lt;300s</div>
            <div class="kpi-icon">⚡</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">{total_hp:,}</div>
            <div class="kpi-label">Honeypots Screened</div>
            <div class="kpi-subtext">{total_hp/st.session_state.total_candidates:.2%} of pool isolated</div>
            <div class="kpi-icon">🛡️</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs layout
    tab_dash, tab_anal, tab_jd, tab_cal, tab_compare = st.tabs([
        "Recruiter Leaderboard",
        "Talent Pool Analytics",
        "Job Requirements Config",
        "Pipeline Calibration",
        "Compare",
    ])
    
    # ── TAB 1: DASHBOARD / LEADERBOARD ────────────────────────────────────────
    with tab_dash:
        col_f, col_l = st.columns([1, 3])
        
        # Sidebar-like Filter Panel
        with col_f:
            st.markdown("<h4 style='margin-top:0;'>🔍 Search & Filters</h4>", unsafe_allow_html=True)
            search_query = st.text_input("Profile Keyword", placeholder="Search title, company, skills, or ID...", label_visibility="collapsed")
            
            st.markdown("---")
            min_score = st.slider("Min Final Match Score", 0.0, 1.0, 0.0, 0.05)
            yoe_min, yoe_max = st.slider("Experience (Years)", 0, 20, (0, 20))
            
            st.markdown("---")
            loc_filter = st.multiselect(
                "Filter Locations", 
                ["Target India Cities", "Other India", "Global"], 
                default=["Target India Cities", "Other India", "Global"]
            )
            
            st.markdown("---")
            hp_filter = st.selectbox(
                "Integrity Filtering",
                ["Exclude Honeypots", "Show Honeypots Only", "Show All Profiles"]
            )
            
            st.markdown("---")
            top_k_show = st.number_input("Candidates Per Page", min_value=5, max_value=100, value=10, step=5)
            
            # Export CSV
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for idx_row, row in enumerate(scored_candidates, 1):
                feat = row["features"]
                # Lazy reasoning generation for CSV exports
                rz_val = row.get("reasoning") or generate_reasoning(feat, row, idx_row)
                writer.writerow([row["candidate_id"], idx_row, f"{row['final_score']:.6f}", rz_val])
                
            st.download_button(
                "⬇️ Download submission.csv",
                data=buf.getvalue(),
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        # Matches list
        with col_l:
            # Apply filters
            filtered_rows = []
            for row in scored_candidates:
                feat = row["features"]
                
                # Search keyword matching
                if search_query:
                    q = search_query.lower()
                    title = (feat.get("current_title") or "").lower()
                    company = (feat.get("current_company") or "").lower()
                    skills_str = " ".join(feat.get("skill_names") or []).lower()
                    cid = row["candidate_id"].lower()
                    if q not in title and q not in company and q not in skills_str and q not in cid:
                        continue
                        
                # Score slider check
                if row["final_score"] < min_score:
                    continue
                    
                # YoE slider check
                yoe = feat.get("years_of_experience")
                if yoe is not None:
                    if yoe < yoe_min or yoe > yoe_max:
                        continue
                        
                # Location multiselect check
                loc = (feat.get("location") or "").lower()
                country = (feat.get("country") or "").lower()
                in_target = any(c in loc for c in config.TARGET_CITIES)
                in_india = country == "india" or any(c in loc for c in config.TARGET_CITIES)
                
                if in_target:
                    loc_cat = "Target India Cities"
                elif in_india:
                    loc_cat = "Other India"
                else:
                    loc_cat = "Global"
                    
                if loc_cat not in loc_filter:
                    continue
                    
                # Honeypot integrity filter
                is_hp = row["is_honeypot_flagged"]
                if hp_filter == "Exclude Honeypots" and is_hp:
                    continue
                elif hp_filter == "Show Honeypots Only" and not is_hp:
                    continue
                    
                filtered_rows.append(row)
                
            total_matched = len(filtered_rows)
            
            st.markdown(f"#### 🏆 Ranked Matches ({total_matched:,} candidates match filters)", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download submission.csv — Top 100 Ready to Submit",
                data=buf.getvalue(),
                file_name="submission.csv",
                mime="text/csv",
                type="primary",
            )
            
            if total_matched == 0:
                st.info("No candidates match your active filters. Try widening search parameters.")
            else:
                # Simple page switcher
                num_pages = max(1, (total_matched + top_k_show - 1) // top_k_show)
                
                col_pag1, col_pag2 = st.columns([7, 3])
                with col_pag2:
                    current_page = st.number_input(f"Page (1 to {num_pages})", min_value=1, max_value=num_pages, value=1, step=1)
                
                start_idx = (current_page - 1) * top_k_show
                end_idx = min(start_idx + top_k_show, total_matched)
                
                # Render current page
                page_candidates = filtered_rows[start_idx:end_idx]
                
                for idx, row in enumerate(page_candidates, start=start_idx + 1):
                    feat = row["features"]
                    
                    # Generate reasoning on display
                    rz = row.get("reasoning")
                    if not rz:
                        rz = generate_reasoning(feat, row, idx)
                        row["reasoning"] = rz
                        
                    is_hp_flag = row["is_honeypot_flagged"]
                    title = feat.get("current_title") or "Role Unspecified"
                    company = feat.get("current_company") or "Company Unspecified"
                    yoe = feat.get("years_of_experience") or "?"
                    loc_raw = (feat.get("location") or "Location Unspecified").title()
                    score_val = row["final_score"]
                    
                    # Match percentage badge styling
                    if score_val >= 0.70:
                        score_style = "badge-score-high"
                    elif score_val >= 0.55:
                        score_style = "badge-score-mid"
                    else:
                        score_style = "badge-score-low"
                        
                    # Disqualifiers and Warning Banners
                    POSITIVE_TRIGGERS = [
                        "strong embeddings", "has ranking-evaluation", "short notice period",
                        "recently active", "evaluation framework experience",
                    ]
                    CONCERN_TRIGGERS = [
                        "little/no", "inactive for", "low recruiter", "not flagged",
                        "long notice", "consulting", "pure research", "senior/architect",
                    ]

                    positive_notes, concern_notes = [], []
                    for note in row["notes"]:
                        if "HONEYPOT" in note:
                            continue
                        if any(t in note.lower() for t in POSITIVE_TRIGGERS):
                            positive_notes.append(note)
                        else:
                            concern_notes.append(note)

                    warnings_box = ""
                    if is_hp_flag:
                        warnings_box += '<div class="warning-banner">🚨 <b>HONEYPOT DETECTED</b>: Excluded from submission.</div>'
                    for note in concern_notes:
                        warnings_box += f'<div class="warning-banner">⚠️ <b>Concern</b>: {note}</div>'
                    for note in positive_notes:
                        warnings_box += (
                            f'<div style="background:rgba(16,185,129,0.08);border:1px solid '
                            f'rgba(16,185,129,0.25);color:#34d399;padding:10px 14px;'
                            f'border-radius:8px;font-size:0.85rem;margin-bottom:10px;">'
                            f'✅ <b>Strength</b>: {note}</div>'
                        )
                        
                    # Matched target keywords
                    matched_words = []
                    search_space_text = f"{feat.get('career_text','') or ''} {feat.get('skill_text','') or ''}".lower()
                    for tag_label, key_terms in [
                        ("Embeddings/Retrieval", config.EMBEDDING_RETRIEVAL_TERMS),
                        ("Vector DB", config.VECTOR_DB_TERMS),
                        ("Ranking Eval", config.EVAL_FRAMEWORK_TERMS),
                        ("Python", config.PYTHON_TERMS)
                    ]:
                        if any(t in search_space_text for t in key_terms):
                            matched_words.append(tag_label)
                            
                    matched_pills = " ".join([
                        f'<span class="badge-meta" style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); color: #a5b4fc;">{label}</span>' 
                        for label in matched_words
                    ])
                    
                    # Basic attributes tags
                    general_pills_list = [
                        f'<span class="badge-meta">&#128198; {yoe} Yrs Exp</span>',
                        f'<span class="badge-meta">&#9679; {loc_raw}</span>',
                    ]
                    notice = feat.get("notice_period_days")
                    if notice is not None:
                        general_pills_list.append(
                            f'<span class="badge-meta" style="color:#fbbf24;border-color:#92400e;">'
                            f'&#9201; {notice}d Notice</span>'
                        )
                    if feat.get("open_to_work"):
                        general_pills_list.append(
                            f'<span class="badge-meta" style="color:#34d399;border-color:#065f46;">'
                            f'&#9679; Open to Work</span>'
                        )
                    general_pills = " ".join(general_pills_list)
                    
                    # Render Main card html
                    card_html_content = f"""
                    <div class="candidate-card" style="margin-bottom:0px; border-bottom-left-radius:0px; border-bottom-right-radius:0px;">
                        <div class="card-header">
                            <div>
                                <div style="display:flex;align-items:center;">
                                    {company_logo_html(company)}
                                    <div>
                                        <div class="card-title">#{idx} | {title}</div>
                                        <div class="card-subtitle">{company}</div>
                                    </div>
                                </div>
                            </div>
                            <div class="badge-score {score_style}">
                                  {score_val:.1%} Match
                            </div>
                        </div>
                        <div class="meta-container">
                            {general_pills}
                            {matched_pills}
                        </div>
                        {warnings_box}
                        <div class="ai-reasoning">
                            💬 <b>Recruiter Summary:</b> {rz}
                        </div>
                    </div>
                    """
                    st.markdown(card_html_content, unsafe_allow_html=True)
                    
                    # Dropdown details area using native Streamlit expander attached cleanly
                    with st.expander("🔍 View Technical Breakdown & Resume History"):
                        st.markdown(f"""
                        <div class="score-grid">
                            <div class="score-grid-item">
                                <span class="score-grid-label">Must-Have Skills</span>
                                <span class="score-grid-val">{row['must_have_score']:.1%}</span>
                            </div>
                            <div class="score-grid-item">
                                <span class="score-grid-label">Semantic Sim.</span>
                                <span class="score-grid-val">{row['semantic_fit_score']:.1%}</span>
                            </div>
                            <div class="score-grid-item">
                                <span class="score-grid-label">Experience Mult.</span>
                                <span class="score-grid-val">{row['experience_fit']:.2f}x</span>
                            </div>
                            <div class="score-grid-item">
                                <span class="score-grid-label">Location Mult.</span>
                                <span class="score-grid-val">{row['location_fit']:.2f}x</span>
                            </div>
                            <div class="score-grid-item">
                                <span class="score-grid-label">Behavioral Mult.</span>
                                <span class="score-grid-val">{row['behavioral_multiplier']:.2f}x</span>
                            </div>
                            <div class="score-grid-item">
                                <span class="score-grid-label">Honeypot Mult.</span>
                                <span class="score-grid-val">{row['honeypot_multiplier']:.2f}x</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_exp1, col_exp2 = st.columns(2)
                        with col_exp1:
                            st.markdown("<p style='font-size:0.95rem; font-weight:600; color:#e2e8f0; margin-bottom:10px;'>💼 Experience Timeline</p>", unsafe_allow_html=True)
                            career = feat.get("career") or []
                            if career:
                                for job in career[:4]:
                                    j_title = job.get("title") or "Role"
                                    j_company = job.get("company") or "Company"
                                    j_duration = job.get("duration_months") or 0
                                    j_desc = job.get("description") or ""
                                    j_desc_clipped = j_desc[:180] + "..." if len(j_desc) > 180 else j_desc
                                    
                                    st.markdown(f"""
                                    <div style="margin-bottom: 12px; border-left: 2px solid #4f46e5; padding-left: 10px;">
                                        <div style="font-weight:600; color:#f1f5f9; font-size:0.85rem;">{j_title}</div>
                                        <div style="color:#94a3b8; font-size:0.75rem; margin-bottom: 3px;">{j_company} • {j_duration} months</div>
                                        <div style="color:#cbd5e1; font-size:0.78rem; line-height:1.4;">{j_desc_clipped}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.caption("No career history details listed.")
                                
                        with col_exp2:
                            st.markdown("<p style='font-size:0.95rem; font-weight:600; color:#e2e8f0; margin-bottom:10px;'>🛠 Skills Inventory</p>", unsafe_allow_html=True)
                            skills = feat.get("skills") or []
                            if skills:
                                skill_tags = []
                                for s in skills[:20]:
                                    s_name = s.get("name") or ""
                                    s_duration = s.get("duration_months") or 0
                                    s_prof = s.get("proficiency") or "Intermediate"
                                    
                                    # Highlight keyword family matches
                                    matched_f = False
                                    for terms_list in (config.EMBEDDING_RETRIEVAL_TERMS + config.VECTOR_DB_TERMS + config.EVAL_FRAMEWORK_TERMS + config.PYTHON_TERMS):
                                        if terms_list in s_name.lower():
                                            matched_f = True
                                            break
                                            
                                    p_class = "tag-skill-matched" if matched_f else "tag-skill-unmatched"
                                    skill_tags.append(f'<span class="tag-skill {p_class}" title="{s_prof}">{s_name} <small style="opacity:0.7">({s_duration}m)</small></span>')
                                    
                                st.markdown(f'<div class="tag-container">{" ".join(skill_tags)}</div>', unsafe_allow_html=True)
                            else:
                                st.caption("No technical skills list listed.")
                                
                            st.markdown("<p style='font-size:0.95rem; font-weight:600; color:#e2e8f0; margin-top:15px; margin-bottom:10px;'>💻 Platform Profile Stats</p>", unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="background:#0f172a; padding: 12px; border-radius: 8px; border:1px solid #1e293b; font-size:0.8rem; line-height:1.6; color:#cbd5e1;">
                                📡 <b>Recruiter Response:</b> {feat.get('recruiter_response_rate', 0):.0%} | 
                                ⏰ <b>Last Active:</b> {feat.get('days_inactive', '?')} days ago<br/>
                                💻 <b>Github activity:</b> {feat.get('github_activity_score', '?')}/5 | 
                                🔗 <b>Connections:</b> {feat.get('connection_count', 0)} connections
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

    # ── TAB 2: ANALYTICS ──────────────────────────────────────────────────────
    with tab_anal:
        st.markdown("### 📊 Talent Pool Statistical Analytics")
        st.markdown("Interactive distributions of profile attributes across the entire loaded candidate dataset.")
        
        # Read pre-calculated aggregates from session state
        yoe_counts = st.session_state.yoe_counts
        loc_counts = st.session_state.loc_counts
        total_hp = st.session_state.total_hp
        hp_counts = st.session_state.hp_counts
        
        # Render charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("#### ⏳ Years of Experience Distribution")
            st.bar_chart(yoe_counts)
            st.caption("Distribution of candidates. The target band is 5-9 years, which is mathematically boosted.")
        with col_chart2:
            st.markdown("#### 🌍 Candidate Location Segmentation")
            st.bar_chart(loc_counts)
            st.caption("Location distribution. Candidates in Target Indian Cities receive maximum location multiplier.")
            
        st.divider()
        
        st.markdown("#### 🛡️ Anti-Honeypot Integrity Scan")
        col_hp_tele1, col_hp_tele2 = st.columns([1, 2])
        with col_hp_tele1:
            st.metric("Total Flagged Honeypots", f"{total_hp:,}", delta=f"{total_hp/len(scored_candidates):.2%} of pool", delta_color="inverse")
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); padding: 16px; border-radius: 10px; font-size: 0.85rem; color:#cbd5e1; line-height: 1.5;">
                ℹ️ <b>Exclusion Engine:</b> A candidate is classified as a honeypot if they trigger 
                <b>2 or more</b> independent checks. Flagged candidates receive a <b>99% score penalty</b>, 
                knocking them out of the leaderboard.
            </div>
            """, unsafe_allow_html=True)
        with col_hp_tele2:
            if not hp_counts.empty:
                st.bar_chart(hp_counts)
                st.caption("Frequencies of integrity scanner rules triggered across the flagged pool.")
            else:
                st.info("No honeypots detected in the loaded dataset.")

    # ── TAB 3: JOB DESCRIPTION INSIGHTS ───────────────────────────────────────
    with tab_jd:
        st.markdown("### 📖 Job Description Configuration & AI Guardrails")
        st.markdown("The parameters parsed from the Job Description document that configure the scoring pipeline rules.")
        
        col_jd1, col_jd2 = st.columns(2)
        with col_jd1:
            st.markdown(f"""
            <div class="landing-card">
                <h4 style="color: #6366f1;"> Target Requirements</h4>
                <ul style="color:#cbd5e1; font-size:0.9rem; line-height:1.7; padding-left:20px;">
                    <li><b>Experience Target:</b> {config.EXPERIENCE_BAND_MIN}-{config.EXPERIENCE_BAND_MAX} Years. Scales down outside (boosted around 6-8 Yrs).</li>
                    <li><b>Primary Locations:</b> Target Cities in India.</li>
                    <li><b>Target Indian Cities:</b> {", ".join(sorted([c.title() for c in config.TARGET_CITIES]))}</li>
                    <li><b>Skills - Embeddings/Retrieval:</b> BGE, Sentence Transformers, Dense retrieval, OpenAI embeddings, Semantic search.</li>
                    <li><b>Skills - Vector Infrastructure:</b> Pinecone, Weaviate, Qdrant, Milvus, Opensearch, Faiss.</li>
                    <li><b>Skills - Evaluation Frameworks:</b> NDCG, MRR, MAP, Offline/Online Evaluation.</li>
                    <li><b>Development Core:</b> Python.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col_jd2:
            st.markdown(f"""
            <div class="landing-card">
                <h4 style="color: #ef4444;"> System Exclusion Guards</h4>
                <ul style="color:#cbd5e1; font-size:0.9rem; line-height:1.7; padding-left:20px;">
                    <li><b>Consulting Carrier Check:</b> Penalizes careers spent exclusively inside massive IT service/consulting companies.</li>
                    <li><b>Academic-Only Flag:</b> Checks for heavy academic research profiles lacking production ML engineering tenure.</li>
                    <li><b>Domain Mismatches:</b> Screens out pure Computer Vision, Robotics, SLAM, or Speech profiles lacking Natural Language context.</li>
                    <li><b>Seniority without Signal:</b> Red-flags senior title holders (Directors, Architects, VPs) who lack software development activity.</li>
                    <li><b>7-Point Integrity Check:</b> Guards against synthetic CV stuffing, connection fraud, and overlapping employment dates.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("####  Original Job Description text")
        st.text_area("Parsed Text Output", value=st.session_state.jd_text, height=300, disabled=True)

    # ── TAB 4: CALIBRATION SIMULATOR ──────────────────────────────────────────
    with tab_cal:
        st.markdown("###  Scoring Pipeline Calibrator")
        st.markdown("Tweak scoring parameters to re-align weights based on changing business priorities. Scores re-calculate in real time.")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("####  Base Fit Scorer Weights")
            c_w_skills = st.slider("Must-Have Skills Weight", 0.0, 1.0, config.WEIGHT_MUST_HAVE_SKILLS, 0.05, key="c_w_skills")
            c_w_sem = st.slider("Semantic Fit Weight", 0.0, 1.0, config.WEIGHT_SEMANTIC_FIT, 0.05, key="c_w_sem")
            
            # Warn if weights don't sum to 1.0
            sum_weights = c_w_skills + c_w_sem
            if abs(sum_weights - 1.0) > 0.01:
                st.warning(f"Weights sum to {sum_weights:.2f}. They will be automatically normalized to 1.0 upon applying.")
                
            st.markdown("####  Heuristic Modifier Floors")
            c_exp_floor = st.slider("Experience Multiplier Floor", 0.1, 1.0, config.EXPERIENCE_MULTIPLIER_FLOOR, 0.05, key="c_exp_floor")
            c_loc_floor = st.slider("Location Multiplier Floor", 0.1, 1.0, config.LOCATION_MULTIPLIER_FLOOR, 0.05, key="c_loc_floor")
            
        with col_c2:
            st.markdown("#### Platform Availability & Integrity")
            c_beh_floor = st.slider("Behavioral Multiplier Floor", 0.1, 1.0, config.BEHAVIORAL_MULTIPLIER_FLOOR, 0.05, key="c_beh_floor")
            c_hp_thresh = st.slider("Honeypot Flag Threshold", 1, 5, config.HONEYPOT_FLAG_THRESHOLD, 1, key="c_hp_thresh")
            
            st.markdown("####  Semantic Match Model")
            if cache_path.exists():
                c_model_engine = st.selectbox(
                    "Semantic Engine Selection",
                    ["BGE Dense Vectors (Deep Semantic, Precomputed)", "TF-IDF + LSA (Bag of Words)"],
                    index=0 if st.session_state.cache_used else 1,
                    key="c_model_engine"
                )
            else:
                st.selectbox("Semantic Engine Selection", ["TF-IDF + LSA (Bag of Words)"], disabled=True, key="c_model_engine")
                c_model_engine = "TF-IDF + LSA (Bag of Words)"

        st.markdown("<br/>", unsafe_allow_html=True)
        
        if st.button("🔄 Apply Calibration & Recalibrate Pool", type="primary", use_container_width=True):
            with st.spinner("Recalibrating composite scorer..."):
                # Normalize weights
                norm_w_skills = c_w_skills
                norm_w_sem = c_w_sem
                if abs(sum_weights - 1.0) > 0.01 and sum_weights > 0:
                    norm_w_skills = c_w_skills / sum_weights
                    norm_w_sem = c_w_sem / sum_weights
                    
                # Update global config state
                config.WEIGHT_MUST_HAVE_SKILLS = norm_w_skills
                config.WEIGHT_SEMANTIC_FIT = norm_w_sem
                config.EXPERIENCE_MULTIPLIER_FLOOR = c_exp_floor
                config.LOCATION_MULTIPLIER_FLOOR = c_loc_floor
                config.BEHAVIORAL_MULTIPLIER_FLOOR = c_beh_floor
                config.HONEYPOT_FLAG_THRESHOLD = c_hp_thresh
                
                # Update model engine preference
                target_cache = "BGE" in c_model_engine
                st.session_state.initial_engine_mode = "BGE Dense Vectors (Precomputed)" if target_cache else "TF-IDF + LSA (Bag of Words)"
                
                st.session_state.has_run = False
                st.success("Calibration applied. Recalibrating pool...")
                time.sleep(0.5)
                st.rerun()

    # ── TAB 5: COMPARE TWO CANDIDATES ──────────────────────────────────────────
    with tab_compare:
        st.markdown("### Compare Two Candidates")
        options = {f"#{i+1} {r['features']['current_title']} — {r['candidate_id']}": r
                   for i, r in enumerate(scored_candidates[:50])}
        col_a, col_b = st.columns(2)
        with col_a:
            choice_a = st.selectbox("Candidate A", list(options.keys()), key="cmp_a")
        with col_b:
            choice_a_idx = list(options.keys()).index(choice_a)
            default_b = min(choice_a_idx + 1, len(options) - 1)
            choice_b = st.selectbox("Candidate B", list(options.keys()),
                                      index=default_b, key="cmp_b")

        a, b = options[choice_a], options[choice_b]
        metrics = [
            ("Final Score", "final_score", "{:.1%}"),
            ("Must-Have Skills", "must_have_score", "{:.1%}"),
            ("Semantic Fit", "semantic_fit_score", "{:.1%}"),
            ("Experience Fit", "experience_fit", "{:.2f}x"),
            ("Location Fit", "location_fit", "{:.2f}x"),
            ("Behavioral Mult.", "behavioral_multiplier", "{:.2f}x"),
        ]
        for label, key, fmt in metrics:
            va, vb = a[key], b[key]
            winner = "A" if va > vb else ("B" if vb > va else "tie")
            c1, c2, c3 = st.columns([1, 2, 1])
            c1.markdown(f"**{fmt.format(va)}**" + (" 🟢" if winner=="A" else ""))
            c2.markdown(f"<center>{label}</center>", unsafe_allow_html=True)
            c3.markdown(f"**{fmt.format(vb)}**" + (" 🟢" if winner=="B" else ""))

else:
    # ── Landing/Welcome Page ──────────────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); border: 1px solid #312e81; border-radius: 14px; padding: 32px; margin-bottom: 28px; text-align: center;">
        <h2 style="font-family:'Outfit', sans-serif; font-weight:800; color:#f8fafc; margin-bottom: 12px; margin-top: 0;"> TalentLens AI</h2>
        <p style="color:#cbd5e1; font-size:1.1rem; max-width: 800px; margin: 0 auto 16px;">
            An AI recruiter that understands candidates, explains every decision, catches suspicious profiles, and finds the best fit through semantic reasoning — not keyword matching.
        </p>
        <span style="display:inline-block; background:rgba(99, 102, 241, 0.15); border:1px solid rgba(99, 102, 241, 0.3); padding:4px 12px; border-radius:6px; font-size:0.85rem; color:#a5b4fc; font-weight:600;">
            👈 Configure sourcing files in the sidebar and click "Run Scoring Pipeline" to begin.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown("""
        <div class="landing-card">
            <div class="landing-icon">🔍</div>
            <h3>Dual Semantic Models</h3>
            <p style="color:#94a3b8; font-size:0.88rem; line-height:1.6;">
                Leverage <b>BGE Dense Vector Embeddings</b> for deep similarity checks or fall back to local 
                <b>TF-IDF + LSA</b> CPU bag-of-words similarity matrices depending on deployment constraints.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_l2:
        st.markdown("""
        <div class="landing-card">
            <div class="landing-icon">⚖️</div>
            <h3>Recruiter-Calibrated</h3>
            <p style="color:#94a3b8; font-size:0.88rem; line-height:1.6;">
                Avoid simple keyword matching. Experience ranges, target cities, IT service firm penalties, and product company indicators are modeled as multiplicative scoring multipliers.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_l3:
        st.markdown("""
        <div class="landing-card">
            <div class="landing-icon">🛡️</div>
            <h3>Honeypot Integrity Scan</h3>
            <p style="color:#94a3b8; font-size:0.88rem; line-height:1.6;">
                Automatically screen synthetic candidate stuffing. Runs 7 independent, explainable checks (career overlapping, connection ratios, Broad Expert profile empty of data) to filter fraud.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("Pipeline Methodology Specifications")
    
    st.markdown("""
    | Scorer Component | Evaluation Method | Objective |
    | :--- | :--- | :--- |
    | **Must-Have Skills (55%)** | Regex match + Duration/Endorsement depth checks | Confirms production experience in Embeddings, Vector DBs, Python, and IR evaluations. Mentions in career history weighted **3× over list tags**. |
    | **Semantic Fit (45%)** | BGE Cosine / TF-IDF + LSA matrix calculation | Evaluates overall career summary and experience text similarity to JD. Matches "right skills, wrong buzzwords" candidates. |
    | **Experience Multiplier** | Triangular function (Peak: 6-8 Yrs) | Penalizes candidates under 5 Yrs or over 9 Yrs, but allows high-skill candidates to pass (soft floor filter). |
    | **Location Multiplier** | Geography check (Pune, Noida, Mumbai, etc.) | Targets maximum alignment for candidates based in target Indian cities, flexible elsewhere. |
    | **IT Services Excluder** | Red Flag checking | Screens candidates whose entire career is inside major IT services consultancies. |
    | **Honeypot Anomaly Filter** | Multi-rule anomaly detector | Catches impossible profiles (e.g. 10 years experience claimed but only 3 months in jobs list, endorsements exceeding connections 4×). |
    """)


