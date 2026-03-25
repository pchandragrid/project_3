import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "in",
    "on",
    "at",
    "to",
    "of",
    "and",
    "with",
    "for",
    "from",
    "by",
    "this",
    "that",
    "it",
    "its",
    "as",
    "be",
    "was",
    "were",
    "has",
    "have",
    "had",
    "into",
    "over",
    "under",
    "near",
}


def load_jsonl(path: str, limit: int = 0) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]


def unique_ngram_ratio(captions: Sequence[str], n_values: Sequence[int] = (1, 2, 3)) -> float:
    all_ngrams: List[Tuple[str, ...]] = []
    for cap in captions:
        toks = tokenize(cap)
        for n in n_values:
            all_ngrams.extend(ngrams(toks, n))
    if not all_ngrams:
        return 0.0
    uniq = len(set(all_ngrams))
    return float(uniq / len(all_ngrams))


def classify_style(caption: str) -> str:
    toks = tokenize(caption)
    length = len(toks)
    detailed_markers = {
        "wearing",
        "holding",
        "standing",
        "sitting",
        "riding",
        "next",
        "behind",
        "while",
        "through",
        "under",
        "near",
        "colorful",
        "large",
        "small",
        "young",
        "old",
    }
    marker_hits = sum(1 for t in toks if t in detailed_markers)

    if length <= 8:
        return "short"
    if length >= 16:
        return "long"
    if length >= 12 and marker_hits >= 2:
        return "detailed"
    return "other"


def top_keywords(captions: Iterable[str], top_k: int = 10) -> List[Tuple[str, int]]:
    cnt = Counter()
    for cap in captions:
        for tok in tokenize(cap):
            if tok in STOPWORDS or len(tok) < 3:
                continue
            cnt[tok] += 1
    return cnt.most_common(top_k)


def cosine_pairwise_diversity(vectors: np.ndarray) -> float:
    """
    Mean pairwise cosine distance for shape [N, D].
    """
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    vn = vectors / norms
    sims = vn @ vn.T
    n = sims.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = 1.0 - sims[mask]
    return float(np.mean(vals)) if vals.size > 0 else 0.0


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

