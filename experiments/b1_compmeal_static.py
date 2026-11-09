"""G2: CompMeal on the non-temporal clean protocols, P1 (mask) and cold_p1 (cold meals, masked).

On P2, CompMeal learns from time windows before the cutoff (b1_compmeal.py). P1 and cold_p1
have no cutoff, so training simulates the protocol itself inside the training data:
- P1: hold out a random 20% of training pairs, apply the P1 mask to them (drop the user's
  course edges of a held-out meal unless a remaining training meal explains them), compute
  features from what is left, and label the held-out pairs.
- cold_p1: hold out a random 20% of training MEALS with all their pairs, mask the same way.
  Meal-popularity features are dropped: every cold meal has zero training popularity, so
  they would only teach the model to spot the split.
Three such folds give the training rows. The number of trees is chosen by full-ranking tune
R@20 (features from the full training data) over a fixed grid, the same rule as for every
other model. Features use reviews up to the last review date.
Rows, model and tree-count selection follow mealrec.compmeal_train (shared with P2).

    python -m experiments.b1_compmeal_static

Per-user test R@20 of each seed goes to results/runs_b1/<dataset>/<protocol>/ for paired tests.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import wilcoxon

from experiments.b1_compmeal import PER_USER
from mealrec.baselines import cover_heuristic, train_cover
from mealrec.compmeal import catalog, features_at, reviews_table
from mealrec.compmeal_train import fit_select, sample_rows
from mealrec.data import DATASETS, load
from mealrec.meta import user_course_meta
from mealrec.metrics import mask_scores, per_user_metrics, ranking_metrics, topk
from mealrec.protocols import completion_times, get, mask_heldout

OUT = Path(__file__).resolve().parent.parent / "results" / "b1_compmeal_static.json"
PROTOCOLS = ("p1", "cold_p1")
N_FOLDS = 3
HOLDOUT = 0.2
SEEDS = (0, 1, 2, 3, 4)
DROP = {"p1": (), "cold_p1": ("meal/log_pop", "meal/log_pop_recent")}


def held_out_fold(protocol: str, train: sp.csr_matrix, rng: np.random.Generator) -> sp.csr_matrix:
    """A random 20% of training pairs (p1) or of training meals with all their pairs (cold_p1)."""
    if protocol == "p1":
        u, m = train.nonzero()
        pick = rng.random(len(u)) < HOLDOUT
        return sp.csr_matrix((np.ones(pick.sum(), np.float32), (u[pick], m[pick])), shape=train.shape)
    meals = np.flatnonzero(train.getnnz(axis=0))
    chosen = np.zeros(train.shape[1], np.float32)
    chosen[rng.choice(meals, size=int(HOLDOUT * len(meals)), replace=False)] = 1
    out = sp.csr_matrix(train @ sp.diags(chosen))
    out.eliminate_zeros()
    return out


def features(data, known: sp.csr_matrix, uc: pd.DataFrame, cat, comps: pd.DataFrame, t) -> dict:
    """Features with `known` as the user's known meals and data.user_course as the visible reviews."""
    kept = pd.MultiIndex.from_arrays(known.nonzero())
    c = comps.set_index(["u", "m"]).loc[lambda d: d.index.isin(kept)].reset_index()
    feats, k = features_at(t, data, cat, reviews_table(data, uc), c)
    assert (k != known).nnz == 0, "known meals must be exactly the given training pairs"
    return feats


def per_user_r20(scores, data) -> tuple[np.ndarray, np.ndarray]:
    train, test = data.user_meal["train"], data.user_meal["test"]
    users = np.flatnonzero(test.getnnz(axis=1))
    return users, per_user_metrics(topk(mask_scores(scores[users], train[users]), 80), test[users])["recall@20"]


def main() -> None:
    result = {}
    for name in DATASETS:
        base = load(name)
        uc = user_course_meta(name)
        cat = catalog(base)
        comps = completion_times(base, uc).drop_duplicates(["u", "m"])
        t_end = np.datetime64(uc["t"].max())
        for protocol in PROTOCOLS:
            t_start = time.time()
            data = get(protocol, name)
            train, tune, test = (data.user_meal[s] for s in ("train", "tune", "test"))
            rng = np.random.default_rng(0)
            X, Y, G = [], [], []
            for fold in range(N_FOLDS):
                held = held_out_fold(protocol, train, rng)
                known = sp.csr_matrix(train - held)
                known.eliminate_zeros()
                pseudo = mask_heldout(replace(data, user_meal={"train": known, "tune": held, "test": held * 0}))
                feats = features(pseudo, known, uc, cat, comps, t_end)
                x, y, g, names = sample_rows(feats, known, held, rng)
                X.append(x); Y.append(y); G += g
                print(f"{name} {protocol} fold {fold}: {len(g)} users, {len(y)} rows, {y.sum()} positives", flush=True)
            X, Y = np.concatenate(X), np.concatenate(Y)
            feats_full = features(data, train, uc, cat, comps, t_end)

            cols = [i for i, n in enumerate(names) if n not in DROP[protocol]]
            vnames = [names[i] for i in cols]
            per_seed, pu = [], []
            for seed in SEEDS:
                s, trees, curve, model = fit_select(X[:, cols], Y, G, feats_full, vnames, tune, train,
                                                    base.n_users, seed)
                per_seed.append({"tune": ranking_metrics(s, tune, exclude=train),
                                 "test": ranking_metrics(s, test, exclude=train), "trees": trees,
                                 "tune_recall@20_by_trees": curve})
                users, r = per_user_r20(s, data)
                pu.append(r)
                (PER_USER / name / protocol).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(PER_USER / name / protocol / f"seed{seed}.npz", users=users, recall_at_20=r)
                if seed == 0:
                    gain = model.booster_.feature_importance("gain")
                    importance = dict(sorted(zip(vnames, (gain / gain.sum()).round(4).tolist()), key=lambda kv: -kv[1]))

            baselines = {b: ranking_metrics(fn(data), test, exclude=train)
                         for b, fn in (("TrainCover", train_cover), ("CoverHeuristic", cover_heuristic))}
            r = np.array([p["test"]["recall@20"] for p in per_seed])
            n = np.array([p["test"]["ndcg@20"] for p in per_seed])
            cm = np.mean(pu, axis=0)
            tests = {}
            for b, fn in (("TrainCover", train_cover), ("CoverHeuristic", cover_heuristic)):
                bu, br = per_user_r20(fn(data), data)
                assert (bu == users).all()
                d = cm - br
                tests[b] = {"mean_diff": float(d.mean()), "share_users_better": float((d > 0).mean()),
                            "share_users_worse": float((d < 0).mean()),
                            "wilcoxon_p": float(wilcoxon(cm, br).pvalue) if np.any(d) else 1.0}
            result.setdefault(name, {})[protocol] = {
                "compmeal": {"test_recall@20_mean": float(r.mean()), "test_recall@20_sd": float(r.std(ddof=1)),
                             "test_ndcg@20_mean": float(n.mean()), "test_ndcg@20_sd": float(n.std(ddof=1)),
                             "seeds": per_seed},
                "baselines": baselines, "paired_vs_baselines": tests, "features": vnames,
                "importance_gain": importance, "seconds": time.time() - t_start,
            }
            print(f"{name} {protocol} CompMeal test R@20 {r.mean():.4f}±{r.std(ddof=1):.4f}  N@20 {n.mean():.4f}"
                  f"  (trees {[p['trees'] for p in per_seed]})", flush=True)
            for b, m in baselines.items():
                print(f"{name} {protocol} {b:<14} test R@20 {m['recall@20']:.4f}  N@20 {m['ndcg@20']:.4f}"
                      f"  | CompMeal - {b}: {tests[b]['mean_diff']:+.4f}, p = {tests[b]['wilcoxon_p']:.2g}", flush=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
