"""B1 prototype: CompMeal, a compositional ranker, on the temporal protocol P2.

The bar on P2 is not CrossCBR but the lookup baselines (CoverHeuristic R@20 0.128 on H,
0.216 on L; D3). CompMeal scores a meal from per-course signals (mealrec.compmeal)
with a LightGBM LambdaRank model.

Training mimics the test situation without touching it: for windows (t0, t1] before the
P2 cutoff, features are computed from data up to t0 and the labels are the meals
completed in (t0, t1]; meals already completed by t0 are excluded, as training meals
are at test time. Rows, model and tree-count selection (full-ranking tune R@20 at the
cutoff) follow mealrec.compmeal_train, shared with b1_compmeal_static (P1, cold_p1).
Evaluation is the usual full ranking (mealrec.metrics) over all meals not in P2 train.

    python -m experiments.b1_compmeal [--variants full no_cf ...]

The full model's per-user test R@20 is saved to results/runs_b1/ for paired tests.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mealrec.baselines import cover_heuristic, train_cover
from mealrec.compmeal import catalog, features_at, reviews_table
from mealrec.compmeal_train import fit_select, sample_rows
from mealrec.data import DATASETS, load
from mealrec.meta import user_course_meta
from mealrec.metrics import mask_scores, per_user_metrics, ranking_metrics, topk
from mealrec.protocols import completion_times, temporal

OUT = Path(__file__).resolve().parent.parent / "results" / "b1_compmeal.json"
PER_USER = Path(__file__).resolve().parent.parent / "results" / "runs_b1"
WINDOWS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8))  # quantiles of completion time; P2 trains up to 0.8
SEEDS = (0, 1, 2, 3, 4)
TIME_FEATURES = ("log_days_since", "pop_recent", "reviews_recent", "days_since_last")
ABLATIONS = {
    "full": lambda name: True,
    "no_cf": lambda name: not name.endswith("/cf"),
    "no_time": lambda name: not any(k in name for k in TIME_FEATURES),
    "coverage_only": lambda name: name.endswith(("/reviewed", "/in_known_meals", "n_reviewed")),
}


def save_per_user(scores, data, name, seed) -> None:
    train, test = data.user_meal["train"], data.user_meal["test"]
    users = np.flatnonzero(test.getnnz(axis=1))
    pu = per_user_metrics(topk(mask_scores(scores[users], train[users]), 80), test[users])
    (PER_USER / name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PER_USER / name / f"seed{seed}.npz", users=users, **pu)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(ABLATIONS), choices=list(ABLATIONS))
    args = p.parse_args()
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}
    result = {}
    for name in DATASETS:
        t_start = time.time()
        base = load(name)
        uc = user_course_meta(name)
        p2 = temporal(base, uc=uc)
        cat = catalog(base)
        reviews = reviews_table(base, uc)
        comps = completion_times(base, uc)
        cutoff = np.datetime64(p2.cutoffs["train_until"])
        train, tune, test = (p2.user_meal[s] for s in ("train", "tune", "test"))

        rng = np.random.default_rng(0)
        X, Y, G = [], [], []
        for q0, q1 in WINDOWS:
            t0, t1 = (np.datetime64(comps["t"].quantile(q)) for q in (q0, q1))
            feats, known = features_at(t0, base, cat, reviews, comps)
            sel = comps[(comps["t"] > t0) & (comps["t"] <= t1)]
            labels = (known * 0).tolil()
            labels[sel["u"].to_numpy(), sel["m"].to_numpy()] = 1
            x, y, g, names = sample_rows(feats, known, labels.tocsr(), rng)
            X.append(x); Y.append(y); G += g
            print(f"{name} window ({q0}, {q1}]: {len(g)} users, {len(y)} rows, {y.sum()} positives", flush=True)
        X, Y = np.concatenate(X), np.concatenate(Y)

        feats_c, known_c = features_at(cutoff, base, cat, reviews, comps)
        assert (known_c != train).nnz == 0, "features at the cutoff must know exactly the P2 training meals"

        baselines = {b: {s: ranking_metrics(fn(p2), m, exclude=train) for s, m in (("tune", tune), ("test", test))}
                     for b, fn in (("CoverHeuristic", cover_heuristic), ("TrainCover", train_cover))}
        block = {"baselines": baselines, "cutoffs": p2.cutoffs, "features": names, "variants": {}}
        for variant in args.variants:
            keep = ABLATIONS[variant]
            cols = [i for i, n in enumerate(names) if keep(n)]
            vnames = [names[i] for i in cols]
            per_seed = []
            for seed in SEEDS if variant == "full" else SEEDS[:3]:
                s, trees, curve, model = fit_select(X[:, cols], Y, G, feats_c, vnames, tune, train, base.n_users, seed)
                row = {"tune": ranking_metrics(s, tune, exclude=train),
                       "test": ranking_metrics(s, test, exclude=train), "trees": trees,
                       "tune_recall@20_by_trees": curve}
                if variant == "full":
                    save_per_user(s, p2, name, seed)
                if variant == "full" and seed == 0:
                    gain = model.booster_.feature_importance("gain")
                    block["importance_gain"] = dict(sorted(zip(vnames, (gain / gain.sum()).round(4).tolist()),
                                                           key=lambda kv: -kv[1]))
                per_seed.append(row)
            r = np.array([p["test"]["recall@20"] for p in per_seed])
            n = np.array([p["test"]["ndcg@20"] for p in per_seed])
            block["variants"][variant] = {"test_recall@20_mean": float(r.mean()), "test_recall@20_sd": float(r.std(ddof=1)),
                                          "test_ndcg@20_mean": float(n.mean()), "seeds": per_seed}
            print(f"{name} {variant:<14} test R@20 {r.mean():.4f}±{r.std(ddof=1):.4f}  N@20 {n.mean():.4f}  "
                  f"(trees {[p['trees'] for p in per_seed]})", flush=True)
        for b, m in baselines.items():
            print(f"{name} {b:<14} test R@20 {m['test']['recall@20']:.4f}  N@20 {m['test']['ndcg@20']:.4f}")
        block["seconds"] = time.time() - t_start
        # Keep variants that were not rerun this time.
        old = previous.get(name, {}).get("variants", {})
        block["variants"] = {**{k: v for k, v in old.items() if k not in block["variants"]}, **block["variants"]}
        if "importance_gain" not in block and "importance_gain" in previous.get(name, {}):
            block["importance_gain"] = previous[name]["importance_gain"]
        result[name] = block
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
