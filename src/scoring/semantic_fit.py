"""
Semantic fit component.

Computes how semantically close each candidate's profile text is to the JD,
using TF-IDF + Truncated SVD (LSA) rather than a neural embedding model.

Why TF-IDF/LSA instead of sentence-transformers:
  - Zero network calls and zero GPU, by construction -- satisfies the
    compute constraints with no asterisks.
  - Fully reproducible from a pip-installable dependency (scikit-learn),
    no model weights to download/cache/version.
  - On 100k short professional-bio documents, LSA over a well-tuned TF-IDF
    matrix captures topical overlap (e.g. "recommendation system", "ranking",
    "retrieval") reasonably well without needing transformer embeddings,
    and runs in seconds on a single CPU core.
  - This is the component that's supposed to catch the JD's "Tier 5
    candidate who never wrote RAG or Pinecone but built a recommendation
    system at a product company" case -- TF-IDF over n-grams plus LSA's
    co-occurrence smoothing covers a meaningful chunk of this without
    needing dense embeddings.

If you have network access in your own dev environment and want to swap in
a real embedding model (e.g. sentence-transformers/all-MiniLM-L6-v2),
precompute the embeddings offline and cache them to disk -- the *ranking
step* itself must still run with no network, per the submission spec.
See README.md for how to wire that in.
"""

import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#./-]{1,}")


def _tokenize(text: str):
    return _WORD_RE.findall((text or "").lower())


def fit_semantic_space(jd_text: str, all_candidate_texts: list[str], n_components: int = 150):
    """
    Fit one TF-IDF vectorizer + SVD over the JD + entire candidate pool, so
    the JD and every candidate live in the same vector space. Returns the
    fitted vectorizer, the SVD model, and the JD's reduced vector.
    """
    corpus = [jd_text] + all_candidate_texts
    vectorizer = TfidfVectorizer(
        tokenizer=_tokenize,
        lowercase=False,  # already lowercased by _tokenize
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    n_components = min(n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)

    jd_vector = reduced[0:1]
    candidate_vectors = reduced[1:]
    return vectorizer, svd, jd_vector, candidate_vectors


def score_semantic_fit_batch(jd_vector: np.ndarray, candidate_vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between the JD vector and every candidate vector, scaled to [0, 1]."""
    sims = cosine_similarity(jd_vector, candidate_vectors)[0]
    # cosine sim on LSA vectors can go slightly negative; clip and it's
    # already roughly in [-1, 1] -> rescale to [0, 1]
    sims = np.clip(sims, -1.0, 1.0)
    return (sims + 1.0) / 2.0
