"""D4: do trained models rely on the course-coverage shortcut?

For every run on the original split (tune-selected checkpoint):
1. Association: within each user, how much of the variance of the model's meal scores is
   explained by the number of the meal's courses in user_course.txt (0..3)? Reported as
   the pooled correlation ratio eta^2 (variance between coverage levels / total, within
   user) and the mean within-user Spearman rho over candidate (non-train) meals.
2. Counterfactual: keep the trained weights, swap the graph for the P1 graph (leaked
   user-course edges of held-out meals removed) and re-score. The drop in test R@20 is
   what the model owed to the leaked edges. LightGCN never reads user_course: control.

Runs on CPU so it does not compete with training jobs for GPUs.

    python -m experiments.d4_shortcut
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.train_bundle import all_scores, build, parse
from mealrec.baselines import course_coverage
from mealrec.data import DATASETS
from mealrec.metrics import ranking_metrics
from mealrec.protocols import get
from mealrec.runs import MODELS, run_dir, scores, summaries

OUT = Path(__file__).resolve().parent.parent / "results" / "d4_shortcut.json"


def association(s: np.ndarray, cov: np.ndarray, cand: np.ndarray, users: np.ndarray) -> dict:
    ss_between = ss_total = 0.0
    rhos = []
    for u in users:
        x, c = s[u, cand[u]].astype(np.float64), cov[u, cand[u]]
        dev = x - x.mean()
        ss_total += float(dev @ dev)
        for level in np.unique(c):
            sel = c == level
            ss_between += sel.sum() * (x[sel].mean() - x.mean()) ** 2
        if len(np.unique(c)) > 1:
            rhos.append(spearmanr(x, c).statistic)
    return {"eta2_pooled": ss_between / ss_total, "spearman_mean": float(np.mean(rhos))}


def counterfactual(model: str, dataset: str, seed: int, p1) -> np.ndarray:
    a = parse(["--model", model, "--dataset", dataset, "--seed", str(seed)])
    ad = build(model, p1, a, torch.device("cpu"))
    state = torch.load(run_dir("orig", dataset, model, seed) / "model_tune_best.pt", map_location="cpu")
    ad.model.load_state_dict(state, strict=False)  # graph tensors are rebuilt from p1, weights reused
    return all_scores(ad, p1.n_users, torch.device("cpu")).numpy()


def main() -> None:
    torch.set_num_threads(16)
    runs = summaries("orig")
    result = {}
    for dataset in DATASETS:
        data, p1 = get("orig", dataset), get("p1", dataset)
        cov = course_coverage(data).astype(np.int64)
        train, test = data.user_meal["train"], data.user_meal["test"]
        cand = ~(train.toarray() > 0)
        users = np.flatnonzero(test.getnnz(axis=1))
        for model in MODELS:
            seeds = sorted(runs.query("dataset == @dataset and model == @model")["seed"])
            rows = []
            for seed in seeds:
                s = scores("orig", dataset, model, seed)
                row = {"seed": int(seed), **association(s, cov, cand, users),
                       "recall@20": ranking_metrics(s, test, exclude=train)["recall@20"]}
                cf = counterfactual(model, dataset, seed, p1)
                row["recall@20_p1_graph"] = ranking_metrics(cf, test, exclude=train)["recall@20"]
                row["relative_drop"] = 1 - row["recall@20_p1_graph"] / row["recall@20"]
                rows.append(row)
            if rows:
                mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "seed"}
                result.setdefault(dataset, {})[model] = {"mean": mean, "seeds": rows}
                print(f"{dataset} {model:<9} n={len(rows)} eta2 {mean['eta2_pooled']:.3f} rho {mean['spearman_mean']:.3f} "
                      f"R@20 {mean['recall@20']:.4f} -> {mean['recall@20_p1_graph']:.4f} ({mean['relative_drop']:+.1%})",
                      flush=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
