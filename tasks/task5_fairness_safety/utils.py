import json
import os
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def load_jsonl(path: str, limit: int = 0) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def mean_bleu4(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    smooth = SmoothingFunction().method1
    vals = []
    for p, refs in zip(predictions, references):
        pt = tokenize(p)
        rt = [tokenize(r) for r in refs if r]
        if not pt or not rt:
            vals.append(0.0)
            continue
        vals.append(
            float(
                sentence_bleu(
                    rt,
                    pt,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smooth,
                )
            )
        )
    return float(np.mean(vals)) if vals else 0.0


def mean_cider(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    try:
        from pycocoevalcap.cider.cider import Cider
    except Exception:
        return _cider_proxy(predictions, references)

    scorer = Cider()
    gts, res = {}, {}
    for i, (pred, refs) in enumerate(zip(predictions, references)):
        gts[i] = list(refs) if refs else [""]
        res[i] = [pred]
    try:
        score, _ = scorer.compute_score(gts, res)
        score = float(score)
        if np.isnan(score):
            return _cider_proxy(predictions, references)
        return score
    except Exception:
        return _cider_proxy(predictions, references)


def _cider_proxy(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    vals = []
    for pred, refs in zip(predictions, references):
        pt = tokenize(pred)
        if not pt or not refs:
            vals.append(0.0)
            continue
        best = 0.0
        for r in refs:
            rt = tokenize(r)
            if not rt:
                continue
            ns = []
            for n in (1, 2, 3, 4):
                png = [tuple(pt[i : i + n]) for i in range(max(0, len(pt) - n + 1))]
                rng = [tuple(rt[i : i + n]) for i in range(max(0, len(rt) - n + 1))]
                if not png or not rng:
                    ns.append(0.0)
                    continue
                rs = set(rng)
                ov = sum(1 for g in png if g in rs)
                ns.append(ov / max(1, len(png)))
            best = max(best, float(np.mean(ns)))
        vals.append(best)
    return float(np.mean(vals) * 10.0) if vals else 0.0

