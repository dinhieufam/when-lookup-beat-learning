"""Full-ranking accuracy metrics and the MealRec+ healthiness metrics.

Accuracy follows the CrossCBR/MealRec+ convention: every meal not seen in
training is a candidate; Recall@K = hits / |test positives|; NDCG@K uses an
ideal DCG over min(|test positives|, K). Averages are over users with at least
one test positive.

Healthiness follows healthiness_eval.py in WUT-IDEA/MealRecPlus.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def mask_scores(scores: np.ndarray, exclude: sp.csr_matrix | None) -> np.ndarray:
    """Return a float copy of scores with excluded (user, meal) pairs set to -inf."""
    out = np.array(scores, dtype=np.float64, copy=True)
    if exclude is not None:
        rows, cols = exclude.nonzero()
        out[rows, cols] = -np.inf
    return out


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first (ties broken by lower index)."""
    # A full stable sort: argpartition picks among tied items in a numpy-version-dependent way.
    k = min(k, scores.shape[1])
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def per_user_metrics(top: np.ndarray, truth: sp.csr_matrix, ks: tuple[int, ...] = (20, 40, 80)) -> dict[str, np.ndarray]:
    """Recall@K and NDCG@K per row of `top` (rows of `truth` must align and be non-empty)."""
    t = truth.toarray() > 0
    n_pos = t.sum(axis=1)
    hits = np.take_along_axis(t, top, axis=1)
    discount = 1.0 / np.log2(np.arange(2, top.shape[1] + 2))
    out = {}
    for k in ks:
        h = hits[:, :k]
        out[f"recall@{k}"] = h.sum(axis=1) / n_pos
        idcg = np.cumsum(discount[:k])[np.minimum(n_pos, k) - 1]
        out[f"ndcg@{k}"] = (h * discount[:k]).sum(axis=1) / idcg
    return out


def ranking_metrics(
    scores: np.ndarray,
    truth: sp.csr_matrix,
    exclude: sp.csr_matrix | None = None,
    ks: tuple[int, ...] = (20, 40, 80),
) -> dict[str, float]:
    users = np.flatnonzero(truth.getnnz(axis=1))
    s = mask_scores(scores[users], exclude[users] if exclude is not None else None)
    per_user = per_user_metrics(topk(s, max(ks)), truth[users], ks)
    return {m: float(v.mean()) for m, v in per_user.items()}


def health_metrics(
    top: np.ndarray,
    users: np.ndarray,
    meal_fsa: np.ndarray,
    meal_who: np.ndarray,
    user_fsa: np.ndarray,
    user_who: np.ndarray,
) -> dict[str, float]:
    """Mean FSA/WHO of the top-K lists, their gap to each user's history, and ExpoGap.

    ExpoGap splits meals into the healthier and less healthy halves by FSA and
    compares their 1/rank exposure: (A - B) / (A + B). Positive means healthier
    meals get more exposure.
    """
    healthier = np.zeros(len(meal_fsa), dtype=bool)
    healthier[np.argsort(meal_fsa, kind="stable")[: len(meal_fsa) // 2]] = True
    utility = 1.0 / np.arange(1, top.shape[1] + 1)
    in_a = healthier[top]
    a = float((utility * in_a).sum())
    b = float((utility * ~in_a).sum())
    return {
        "fsa_mean": float(meal_fsa[top].mean()),
        "who_mean": float(meal_who[top].mean()),
        "fsa_delta": float((meal_fsa[top].mean(axis=1) - user_fsa[users]).mean()),
        "who_delta": float((meal_who[top].mean(axis=1) - user_who[users]).mean()),
        "expo_gap": (a - b) / (a + b),
    }
