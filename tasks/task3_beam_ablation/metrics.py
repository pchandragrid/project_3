from typing import Dict, List, Sequence, Tuple

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def bleu4_mean(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    smooth = SmoothingFunction().method1
    scores = []
    for pred, refs in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = [r.lower().split() for r in refs if r]
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        score = sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth,
        )
        scores.append(float(score))
    return float(np.mean(scores)) if scores else 0.0


def _lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l_mean(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    scores = []
    for pred, refs in zip(predictions, references):
        pred_tokens = pred.lower().split()
        if not pred_tokens or not refs:
            scores.append(0.0)
            continue

        best_f = 0.0
        for ref in refs:
            ref_tokens = ref.lower().split()
            if not ref_tokens:
                continue
            lcs = _lcs_len(pred_tokens, ref_tokens)
            precision = lcs / max(1, len(pred_tokens))
            recall = lcs / max(1, len(ref_tokens))
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = (2 * precision * recall) / (precision + recall)
            if f1 > best_f:
                best_f = f1
        scores.append(best_f)
    return float(np.mean(scores)) if scores else 0.0


def cider_mean(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    try:
        from pycocoevalcap.cider.cider import Cider
    except Exception:
        return cider_proxy_mean(predictions, references)

    scorer = Cider()
    gts: Dict[int, List[str]] = {}
    res: Dict[int, List[str]] = {}
    for i, (pred, refs) in enumerate(zip(predictions, references)):
        gts[i] = list(refs) if refs else [""]
        res[i] = [pred]
    try:
        score, _ = scorer.compute_score(gts, res)
        score = float(score)
        if np.isnan(score):
            return cider_proxy_mean(predictions, references)
        return score
    except Exception:
        return cider_proxy_mean(predictions, references)


def meteor_mean(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    # Try COCO METEOR first.
    try:
        from pycocoevalcap.meteor.meteor import Meteor

        scorer = Meteor()
        gts: Dict[int, List[str]] = {}
        res: Dict[int, List[str]] = {}
        for i, (pred, refs) in enumerate(zip(predictions, references)):
            gts[i] = list(refs) if refs else [""]
            res[i] = [pred]
        score, _ = scorer.compute_score(gts, res)
        return float(score)
    except Exception:
        pass

    # Fallback to NLTK meteor.
    try:
        from nltk.translate.meteor_score import meteor_score
    except Exception:
        meteor_score = None

    scores = []
    for pred, refs in zip(predictions, references):
        if not pred or not refs:
            scores.append(0.0)
            continue
        ref_tokens = [r.lower().split() for r in refs if r]
        pred_tokens = pred.lower().split()
        if not ref_tokens or not pred_tokens:
            scores.append(0.0)
            continue
        if meteor_score is not None:
            try:
                scores.append(float(meteor_score(ref_tokens, pred_tokens)))
                continue
            except Exception:
                pass
        scores.append(_meteor_proxy_single(pred_tokens, ref_tokens))
    return float(np.mean(scores)) if scores else 0.0


def cider_proxy_mean(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """
    Lightweight CIDEr-like proxy when pycocoevalcap is unavailable.
    Uses averaged n-gram overlap (1..4) against best matching reference.
    """
    vals = []
    for pred, refs in zip(predictions, references):
        pred_tokens = pred.lower().split()
        if not pred_tokens or not refs:
            vals.append(0.0)
            continue
        best = 0.0
        for ref in refs:
            ref_tokens = ref.lower().split()
            if not ref_tokens:
                continue
            score_n = []
            for n in (1, 2, 3, 4):
                p_ngrams = [tuple(pred_tokens[i : i + n]) for i in range(max(0, len(pred_tokens) - n + 1))]
                r_ngrams = [tuple(ref_tokens[i : i + n]) for i in range(max(0, len(ref_tokens) - n + 1))]
                if not p_ngrams or not r_ngrams:
                    score_n.append(0.0)
                    continue
                r_set = set(r_ngrams)
                overlap = sum(1 for g in p_ngrams if g in r_set)
                score_n.append(overlap / max(1, len(p_ngrams)))
            cand = float(np.mean(score_n))
            if cand > best:
                best = cand
        vals.append(best)
    # Scale to CIDEr-like range for readability.
    return float(np.mean(vals) * 10.0) if vals else 0.0


def _meteor_proxy_single(pred_tokens: List[str], refs_tokens: List[List[str]]) -> float:
    """
    METEOR-like proxy: best reference unigram F-mean with higher recall weight.
    """
    best = 0.0
    pset = pred_tokens
    for rt in refs_tokens:
        if not rt:
            continue
        rset = rt
        common = len(set(pset).intersection(set(rset)))
        if common == 0:
            continue
        precision = common / max(1, len(pset))
        recall = common / max(1, len(rset))
        # Similar emphasis as METEOR on recall.
        fmean = (10 * precision * recall) / max(1e-8, recall + 9 * precision)
        if fmean > best:
            best = float(fmean)
    return best

