"""D5: do models capture the compositional (appetizer + main + dessert) structure?

For every run (tune-selected scores), on every protocol:
1. Additivity: R^2 of the best fit score[u, m] ~ sum_{c in m} beta[u, c] (one shared
   least-squares solve; the constant is in the span because every meal has 3 courses).
   R^2 near 1 means the model scores a meal as the sum of its courses.
2. Recall@20 on subsets of test positives:
   - warm: the meal has training interactions; cold: it has none;
   - novel: no meal with training interactions shares two or more courses with it
     (a combination the model never saw in any form).

    python -m experiments.d5_compositional
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from mealrec.data import DATASETS
from mealrec.metrics import mask_scores, topk
from mealrec.protocols import PROTOCOLS, get
from mealrec.runs import MODELS, scores, summaries

OUT = Path(__file__).resolve().parent.parent / "results" / "d5_compositional.json"


def additive_r2(s: np.ndarray, meal_course: sp.csr_matrix) -> float:
    used = np.flatnonzero(meal_course.getnnz(axis=0))
    x = meal_course[:, used].toarray().astype(np.float64)
    y = s.T.astype(np.float64)  # M x U
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dev = y - y.mean(axis=0, keepdims=True)
    return float(1 - (resid ** 2).sum() / (dev ** 2).sum())


def meal_subsets(data) -> dict[str, np.ndarray]:
    pop = np.asarray(data.user_meal["train"].sum(axis=0)).ravel()
    warm = pop > 0
    shared = (data.meal_course @ data.meal_course.T).toarray()  # courses in common
    np.fill_diagonal(shared, 0)
    novel = ~((shared[:, warm] >= 2).any(axis=1)) & ~warm
    return {"warm": warm, "cold": ~warm, "novel": novel}


def subset_recall(s: np.ndarray, data, subset: np.ndarray, k: int = 20) -> tuple[float, int]:
    train, test = data.user_meal["train"], data.user_meal["test"].toarray() > 0
    t = test & subset[None, :]
    users = np.flatnonzero(t.any(axis=1))
    if len(users) == 0:
        return float("nan"), 0
    top = topk(mask_scores(s[users], train[users]), k)
    hits = np.take_along_axis(t[users], top, axis=1).sum(axis=1)
    return float(np.mean(hits / t[users].sum(axis=1))), int(t.sum())


def main() -> None:
    runs = summaries()
    result = {}
    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            data = get(protocol, dataset)
            subsets = meal_subsets(data)
            test = data.user_meal["test"].toarray() > 0
            info = {name: {"meals": int(m.sum()), "test_positives": int(test[:, m].sum())} for name, m in subsets.items()}
            block = {"subsets": info, "models": {}}
            for model in MODELS:
                seeds = sorted(runs.query("protocol == @protocol and dataset == @dataset and model == @model")["seed"])
                rows = []
                for seed in seeds:
                    s = scores(protocol, dataset, model, seed)
                    row = {"additive_r2": additive_r2(s, data.meal_course)}
                    for name, m in subsets.items():
                        row[f"recall@20/{name}"] = subset_recall(s, data, m)[0]
                    rows.append(row)
                if rows:
                    block["models"][model] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
                    block["models"][model]["n_seeds"] = len(rows)
                    r = block["models"][model]
                    print(f"{dataset} {protocol:<4} {model:<9} additive R2 {r['additive_r2']:.3f}  R@20 warm "
                          f"{r['recall@20/warm']:.4f} cold {r['recall@20/cold']:.4f} novel {r['recall@20/novel']:.4f}",
                          flush=True)
            result.setdefault(dataset, {})[protocol] = block
            print(f"{dataset} {protocol}: subsets {info}")
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
