"""D7: are the differences between models robust?

Per protocol and dataset, with the epoch selected on tune:
1. Seed spread: mean, sd, min and max test R@20 over seeds.
2. Paired tests: per-user test R@20 averaged over seeds; Wilcoxon signed-rank for every
   pair of models, Holm-corrected within each (protocol, dataset).
3. Strata: R@20 by tertile of user training degree, and by tertile of meal training
   popularity (recall over the test positives that fall in each tertile).

    python -m experiments.d7_robustness
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from experiments.d5_compositional import subset_recall
from mealrec.data import DATASETS
from mealrec.protocols import PROTOCOLS, get
from mealrec.runs import MODELS, per_user, scores, summaries

OUT = Path(__file__).resolve().parent.parent / "results" / "d7_robustness.json"


def holm(p: dict) -> dict:
    order = sorted(p, key=p.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * p[k]))
        adj[k] = running
    return adj


def tertiles(x: np.ndarray) -> np.ndarray:
    cuts = np.quantile(x, [1 / 3, 2 / 3])
    return np.digitize(x, cuts)  # 0 low, 1 mid, 2 high


def main() -> None:
    runs = summaries()
    result = {}
    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            g = runs.query("protocol == @protocol and dataset == @dataset")
            if g.empty:
                continue
            data = get(protocol, dataset)
            degree = data.user_meal["train"].getnnz(axis=1)
            pop = np.asarray(data.user_meal["train"].sum(axis=0)).ravel()
            pop_bin = tertiles(pop + np.random.default_rng(0).uniform(0, 1e-6, len(pop)))  # break ties at 0
            block = {"seeds": {}, "strata": {}, "per_user": {}}
            for model in MODELS:
                gm = g[g["model"] == model]
                if gm.empty:
                    continue
                r = gm["tune_sel/recall@20"]
                block["seeds"][model] = {"mean": float(r.mean()), "sd": float(r.std(ddof=1)) if len(r) > 1 else None,
                                         "min": float(r.min()), "max": float(r.max()), "n": int(len(r))}
                pu = [per_user(protocol, dataset, model, s) for s in sorted(gm["seed"])]
                users = pu[0]["users"]
                block["per_user"][model] = (users, np.mean([p["recall@20"] for p in pu], axis=0))
                deg_bin = tertiles(degree[users] + np.random.default_rng(0).uniform(0, 1e-6, len(users)))
                strata = {f"user_degree_t{b}": float(block["per_user"][model][1][deg_bin == b].mean()) for b in range(3)}
                for b in range(3):
                    strata[f"meal_pop_t{b}"] = float(np.nanmean(
                        [subset_recall(scores(protocol, dataset, model, s), data, pop_bin == b)[0] for s in sorted(gm["seed"])]))
                block["strata"][model] = strata

            pvals, diffs = {}, {}
            for a, b in itertools.combinations(block["per_user"], 2):
                (ua, ra), (ub, rb) = block["per_user"][a], block["per_user"][b]
                assert (ua == ub).all()
                key = f"{a} vs {b}"
                diffs[key] = float((ra - rb).mean())
                pvals[key] = float(wilcoxon(ra, rb).pvalue) if np.any(ra != rb) else 1.0
            adj = holm(pvals)
            block["paired"] = {k: {"mean_diff_recall@20": diffs[k], "p": pvals[k], "p_holm": adj[k]} for k in pvals}
            del block["per_user"]
            result.setdefault(dataset, {})[protocol] = block
            print(f"\n== {dataset} {protocol}")
            for k, v in sorted(block["paired"].items(), key=lambda kv: kv[1]["p_holm"]):
                print(f"  {k:<24} diff {v['mean_diff_recall@20']:+.4f}  p_holm {v['p_holm']:.2g}")
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
