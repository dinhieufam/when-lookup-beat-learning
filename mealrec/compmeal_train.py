"""Training recipe shared by every CompMeal experiment (P2, P1, cold_p1).

Training rows come from a simulation of the protocol on training data only (time windows
for P2, held-out folds for P1/cold_p1; see the experiments). For each user with labels:
all labelled meals plus N_NEG negatives drawn uniformly from the user's other unknown
meals, shuffled. A LightGBM LambdaRank model is fit with MAX_TREES trees, and the number
of trees is chosen on a fixed grid by full-ranking tune Recall@20, the rule used for
every other model.

Two earlier choices were biased:
- keeping every partly reviewed meal as a negative (plus 100 random ones) made "no course
  reviewed" look positive whenever held-out positives had masked course edges;
- labelled rows first in each group, with early stopping on LightGBM's sampled NDCG,
  which keeps input order among tied scores, so a 1-tree model looked best.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

from mealrec.metrics import ranking_metrics

N_NEG = 500
MAX_TREES = 1000
TREE_GRID = (10, 25, 50, 100, 200, 400, 700, 1000)


def sample_rows(feats: dict, known: sp.csr_matrix, labels: sp.csr_matrix, rng: np.random.Generator):
    """Rows (X, y, group sizes, feature names) for users with at least one label."""
    lab, kn = labels.toarray() > 0, known.toarray() > 0
    names = list(feats)
    xs, ys, groups = [], [], []
    for u in np.flatnonzero(lab.any(axis=1)):
        free = np.flatnonzero(~kn[u] & ~lab[u])
        idx = rng.permutation(np.concatenate([np.flatnonzero(lab[u]),
                                              rng.choice(free, size=min(N_NEG, len(free)), replace=False)]))
        xs.append(np.stack([feats[n][u, idx] for n in names], axis=1))
        ys.append(lab[u, idx].astype(int))
        groups.append(len(idx))
    return np.concatenate(xs), np.concatenate(ys), groups, names


class Truncated:
    """Predict with the first `trees` trees of a fitted LightGBM model."""

    def __init__(self, model, trees: int):
        self.model, self.trees = model, trees

    def predict(self, x):
        return self.model.predict(x, num_iteration=self.trees)


def full_scores(model, feats: dict, names: list[str], n_users: int, chunk: int = 256) -> np.ndarray:
    """U x M scores from per-(user, meal) feature arrays."""
    out = []
    for s in range(0, n_users, chunk):
        x = np.stack([feats[n][s:s + chunk].reshape(-1) for n in names], axis=1)
        out.append(model.predict(x).reshape(-1, feats[names[0]].shape[1]))
    return np.concatenate(out)


def fit_select(X, Y, G, feats: dict, names: list[str], tune: sp.csr_matrix, train: sp.csr_matrix,
               n_users: int, seed: int):
    """Fit on (X, Y, G) and pick the tree count by full-ranking tune R@20.

    Returns the scores of the selected model (U x M), the tree count, the tune curve and the model.
    """
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=MAX_TREES, learning_rate=0.05,
                           num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, n_jobs=16, verbose=-1)
    model.fit(X, Y, group=G)
    curve, best, best_scores = {}, None, None
    for t in TREE_GRID:
        s = full_scores(Truncated(model, t), feats, names, n_users)
        curve[t] = ranking_metrics(s, tune, exclude=train)["recall@20"]
        if best is None or curve[t] > curve[best]:
            best, best_scores = t, s
    return best_scores, best, curve, model
