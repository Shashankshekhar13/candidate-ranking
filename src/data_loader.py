"""
Load the candidate pool and job description.

Supports both candidates.jsonl and candidates.jsonl.gz transparently, since
the hackathon bundle ships the gzipped file and some environments will keep
it compressed.
"""

import gzip
import json
from pathlib import Path
from typing import Iterator


def load_candidates(path: Path) -> list[dict]:
    """Load every candidate record from a .jsonl or .jsonl.gz file."""
    path = Path(path)
    if not path.exists():
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            path = gz_path
        else:
            raise FileNotFoundError(
                f"Could not find {path} or {gz_path}. "
                "Place candidates.jsonl (or .jsonl.gz) in the data/ directory."
            )

    opener = gzip.open if path.suffix == ".gz" else open
    candidates = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def iter_candidates(path: Path) -> Iterator[dict]:
    """Stream candidates one at a time (memory-friendlier for very large pools)."""
    path = Path(path)
    if not path.exists():
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            path = gz_path
        else:
            raise FileNotFoundError(f"Could not find {path} or {gz_path}.")

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_job_description(path: Path) -> str:
    """Load the JD as plain text. Accepts .md, .txt, or .docx (best-effort)."""
    path = Path(path)
    if path.suffix.lower() == ".docx":
        try:
            # pyrefly: ignore [missing-import]
            import docx  # python-docx
        except ImportError as e:
            raise ImportError(
                "python-docx is required to read .docx job descriptions. "
                "pip install python-docx, or convert the JD to .md/.txt."
            ) from e
        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs)

    return path.read_text(encoding="utf-8")
