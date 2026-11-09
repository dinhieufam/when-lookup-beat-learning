"""G5: BundleComp (mealrec.bundlecomp) on bundle benchmarks under the clean protocol P1.

Same recipe as CompMeal on P1 (b1_compmeal_static): training rows simulate P1 inside the
training data (hold out 20% of training pairs, mask their item edges unless a remaining
training bundle explains them, label them); each labelled user keeps all labels plus N_NEG
uniform negatives, shuffled; LightGBM LambdaRank; the tree count is chosen on a fixed grid
by full-ranking tune R@20 over a sample of tune users; test is full ranking over all bundles
for every test user. Reference models: CrossCBR / MultiCBR trained on P1 with their upstream
configs (results/runs/p1/<dataset>/, train_bundle.py), compared per user (Wilcoxon).

    python -m experiments.g5_bundlecomp [--datasets iFashion ...] [--seeds 0 1 2]
    python -m experiments.g5_bundlecomp --drop cover_ui --tag nocover   # ablation, separate outputs

Seeds already in the output (per-user file and JSON row) are reused, not retrained. Several
processes (one per dataset) may run at once: the JSON is merged under a file lock.

Writes results/g5_bundlecomp.json (merged) and per-user test R@20 to results/runs_g5/<dataset>/.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import time
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp
from scipy.stats import wilcoxon

from experiments.b1_compmeal_static import held_out_fold
from mealrec.bundlecomp import FEATURES, BState, chunk_features
from mealrec.compmeal_train import TREE_GRID
from mealrec.data import BUNDLE_DATASETS, DATASETS
from mealrec.metrics import per_user_metrics
from mealrec.protocols import get, mask_heldout
from mealrec.runs import RESULTS, per_user

OUT = RESULTS / "g5_bundlecomp.json"
PER_USER = RESULTS / "runs_g5"
ALL = DATASETS + BUNDLE_DATASETS
N_FOLDS = 2
N_NEG = 300
MAX_USERS_PER_FOLD = 15000
TUNE_USERS = 5000
CHUNK = 512
KS = (20, 40, 80)
FEATS = list(FEATURES)  # features used; --drop removes some


def training_rows(st: BState, known: sp.csr_matrix, labels: sp.csr_matrix, rng, max_users: int):
    users = np.flatnonzero(labels.getnnz(axis=1))
    if len(users) > max_users:
        users = np.sort(rng.choice(users, size=max_users, replace=False))
    xs, ys, groups = [], [], []
    for i in range(0, len(users), CHUNK):
        ub = users[i:i + CHUNK]
        f = chunk_features(st, ub)
        lab = labels[ub].toarray() > 0
        kn = known[ub].toarray() > 0
        for r in range(len(ub)):
            free = np.flatnonzero(~kn[r] & ~lab[r])
            idx = rng.permutation(np.concatenate([np.flatnonzero(lab[r]),
                                                  rng.choice(free, size=min(N_NEG, len(free)), replace=False)]))
            xs.append(np.stack([f[n][r, idx] for n in FEATS], axis=1))
            ys.append(lab[r, idx].astype(int))
            groups.append(len(idx))
    return np.concatenate(xs), np.concatenate(ys), groups


def rank_eval(st: BState, model, trees: list[int], data, split: str, users: np.ndarray) -> dict:
    """Per-user metrics from full ranking over all bundles (training pairs masked), for each
    tree count; features are computed once per user chunk. Returns {trees: {metric: per-user}}."""
    train, truth = data.user_meal["train"], data.user_meal[split]
    out = {t: {f"{m}@{k}": [] for k in KS for m in ("recall", "ndcg")} for t in trees}
    for i in range(0, len(users), CHUNK):
        ub = users[i:i + CHUNK]
        f = chunk_features(st, ub)
        x = np.stack([f[n].reshape(-1) for n in FEATS], axis=1)
        r, c = train[ub].nonzero()
        for t in trees:
            s = model.predict(x, num_iteration=t).reshape(len(ub), -1)
            s[r, c] = -np.inf
            top = np.argpartition(-s, max(KS), axis=1)[:, :max(KS)]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, 1), axis=1, kind="stable"), 1)
            for k, v in per_user_metrics(top, truth[ub], KS).items():
                out[t][k].append(v)
    return {t: {k: np.concatenate(v) for k, v in d.items()} for t, d in out.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(ALL), choices=ALL)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--drop", nargs="*", default=[], choices=FEATURES, help="features to leave out (ablation)")
    p.add_argument("--tag", default="", help="output suffix: results/g5_bundlecomp_<tag>.json, results/runs_g5_<tag>/")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    global OUT, PER_USER
    if a.tag:
        OUT, PER_USER = RESULTS / f"g5_bundlecomp_{a.tag}.json", RESULTS / f"runs_g5_{a.tag}"
    FEATS[:] = [f for f in FEATURES if f not in a.drop]
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in a.datasets:
        t0 = time.time()
        data = get("p1", name)
        train, tune, test = (data.user_meal[s] for s in ("train", "tune", "test"))
        rng = np.random.default_rng(0)
        X, Y, G = [], [], []
        for fold in range(N_FOLDS):
            held = held_out_fold("p1", train, rng)
            known = sp.csr_matrix(train - held)
            known.eliminate_zeros()
            pseudo = mask_heldout(replace(data, user_meal={"train": known, "tune": held, "test": held * 0}))
            x, y, g = training_rows(BState.build(pseudo.user_course, known, data.meal_course), known, held, rng,
                                    MAX_USERS_PER_FOLD)
            X.append(x); Y.append(y); G += g
            print(f"{name} fold {fold}: {len(g)} users, {len(y)} rows, {y.sum()} positives ({time.time() - t0:.0f}s)",
                  flush=True)
        X, Y = np.concatenate(X), np.concatenate(Y)
        st = BState.build(data.user_course, train, data.meal_course)
        tune_users = np.flatnonzero(tune.getnnz(axis=1))
        if len(tune_users) > TUNE_USERS:
            tune_users = np.sort(np.random.default_rng(1).choice(tune_users, TUNE_USERS, replace=False))
        test_users = np.flatnonzero(test.getnnz(axis=1))
        rows = []
        done = {r["seed"]: r for r in result.get(name, {}).get("seeds", [])}
        for seed in a.seeds:
            if seed in done and (PER_USER / name / f"seed{seed}.npz").exists():
                rows.append(done[seed])
                print(f"{name} seed {seed}: reused (test R@20 {done[seed]['test']['recall@20']:.4f})", flush=True)
                continue
            model = lgb.LGBMRanker(objective="lambdarank", n_estimators=max(TREE_GRID), learning_rate=0.05,
                                   num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, random_state=seed, n_jobs=a.threads, verbose=-1)
            model.fit(X, Y, group=G)
            ev = rank_eval(st, model, list(TREE_GRID), data, "tune", tune_users)
            curve = {t: float(ev[t]["recall@20"].mean()) for t in TREE_GRID}
            best = max(curve, key=curve.get)
            pu = rank_eval(st, model, [best], data, "test", test_users)[best]
            (PER_USER / name).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(PER_USER / name / f"seed{seed}.npz", users=test_users, **pu)
            rows.append({"seed": seed, "trees": best, "tune_recall@20_by_trees": curve,
                         "test": {k: float(v.mean()) for k, v in pu.items()}})
            print(f"{name} seed {seed}: trees {best}  tune R@20 {curve[best]:.4f}  test R@20 {rows[-1]['test']['recall@20']:.4f}"
                  f"  N@20 {rows[-1]['test']['ndcg@20']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        ours = np.mean([np.load(PER_USER / name / f"seed{s}.npz")["recall@20"] for s in a.seeds], axis=0)
        block = {"seeds": rows, "test_recall@20_mean": float(np.mean([r["test"]["recall@20"] for r in rows])),
                 "test_recall@20_sd": float(np.std([r["test"]["recall@20"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                 "test_ndcg@20_mean": float(np.mean([r["test"]["ndcg@20"] for r in rows])), "references": {}}
        for ref in ("CrossCBR", "MultiCBR"):
            runs = sorted((RESULTS / "runs" / "p1" / name / ref).glob("seed*/per_user.npz"))
            if not runs:
                continue
            pus = [per_user("p1", name, ref, int(r.parent.name[4:])) for r in runs]
            assert all((x["users"] == test_users).all() for x in pus)
            other = np.mean([x["recall@20"] for x in pus], axis=0)
            d = ours - other
            block["references"][ref] = {"recall@20": float(other.mean()), "n_seeds": len(pus), "mean_diff": float(d.mean()),
                                        "wilcoxon_p": float(wilcoxon(ours, other).pvalue) if np.any(d) else 1.0}
            print(f"{name} vs {ref} (P1, {len(pus)} seeds): {other.mean():.4f}  diff {d.mean():+.4f}  "
                  f"p = {block['references'][ref]['wilcoxon_p']:.2g}", flush=True)
        block["seconds"] = time.time() - t0
        block["features"] = list(FEATS)
        with open(OUT.with_suffix(".lock"), "w") as lock:  # other datasets may be running in parallel
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = json.loads(OUT.read_text()) if OUT.exists() else {}
            result[name] = block
            OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
