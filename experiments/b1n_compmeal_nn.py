"""G3: CompMeal-N (mealrec.compmeal_nn) on P2, P1 and cold_p1, against the GBDT CompMeal.

Training sets simulate each protocol on training data only, exactly as the GBDT runs do:
P2 uses the time windows of b1_compmeal (state at t0, labels = meals completed in (t0, t1]);
P1 / cold_p1 use the held-out folds of b1_compmeal_static. The epoch is chosen by
full-ranking tune R@20; test is read once at that epoch.

    python -m experiments.b1n_compmeal_nn [--cells MealRec+H/p2 ...] [--seeds 0 1 2 3 4] [--variant full]
    python -m experiments.b1n_compmeal_nn --seeds 0 --lr 3e-4 --wd 1e-4 --root runs_b1n_tune   # tuning grid

Writes results/b1n_compmeal_nn.json (merged across invocations) and per-user test R@20 to
results/runs_b1n/<dataset>/<protocol>/<variant>/seed<k>.npz.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from experiments.b1_compmeal import WINDOWS
from experiments.b1_compmeal_static import HOLDOUT, N_FOLDS, held_out_fold  # noqa: F401  (HOLDOUT documents the fold size)
from mealrec.compmeal import catalog, reviews_table, state_at
from mealrec.compmeal_nn import content_embedding, train_compmeal_nn
from mealrec.data import DATASETS, load
from mealrec.meta import course_meta, user_course_meta
from mealrec.metrics import mask_scores, per_user_metrics, topk
from mealrec.protocols import completion_times, get, mask_heldout, temporal

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LR, WD = 3e-4, 1e-4
CELLS = tuple(f"{d}/{p}" for d in DATASETS for p in ("p2", "p1", "cold_p1"))
VARIANTS = {
    "full": {},
    "no_pairs": {"use_pairs": False},
    "no_content": {"use_content": False},
}


def state_for(data, known: sp.csr_matrix, uc: pd.DataFrame, cat, comps: pd.DataFrame, t):
    """State as of t with `known` as the completed meals and data.user_course as visible reviews."""
    kept = pd.MultiIndex.from_arrays(known.nonzero())
    c = comps.set_index(["u", "m"]).loc[lambda d: d.index.isin(kept)].reset_index()
    st = state_at(t, data, cat, reviews_table(data, uc), c)
    assert (st.known != known).nnz == 0, "known meals must be exactly the given training pairs"
    return st


def simulate(protocol: str, base, uc, cat, comps):
    """(training sets [(state, labels)], eval state, protocol data)."""
    if protocol == "p2":
        data = temporal(base, uc=uc)
        reviews = reviews_table(base, uc)
        sets = []
        for q0, q1 in WINDOWS:
            t0, t1 = (np.datetime64(comps["t"].quantile(q)) for q in (q0, q1))
            st = state_at(t0, base, cat, reviews, comps)
            sel = comps[(comps["t"] > t0) & (comps["t"] <= t1)]
            lab = sp.csr_matrix((np.ones(len(sel), np.float32), (sel["u"].to_numpy(), sel["m"].to_numpy())),
                                shape=(base.n_users, base.n_meals))
            sets.append((st, lab))
        ev = state_at(np.datetime64(data.cutoffs["train_until"]), base, cat, reviews, comps)
        assert (ev.known != data.user_meal["train"]).nnz == 0
        return sets, ev, data
    data = get(protocol, base.name)
    train = data.user_meal["train"]
    t_end = np.datetime64(uc["t"].max())
    rng = np.random.default_rng(0)
    sets = []
    for _ in range(N_FOLDS):
        held = held_out_fold(protocol, train, rng)
        known = sp.csr_matrix(train - held)
        known.eliminate_zeros()
        pseudo = mask_heldout(replace(data, user_meal={"train": known, "tune": held, "test": held * 0}))
        sets.append((state_for(pseudo, known, uc, cat, comps, t_end), held))
    return sets, state_for(data, train, uc, cat, comps, t_end), data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+", default=list(CELLS), choices=CELLS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--variant", default="full", choices=list(VARIANTS))
    p.add_argument("--epochs", type=int, default=60)
    # Chosen on seed 0 by tune R@20 summed over the six cells (results/runs_b1n_tune.json):
    # lr {3e-4, 1e-3} x weight decay {1e-5, 1e-4}.
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--wd", type=float, default=WD)
    p.add_argument("--save-scores", action="store_true", help="also save U x M test-time scores (float16)")
    p.add_argument("--root", default="runs_b1n",
                   help="per-user output tree under results/; results JSON is results/<root>.json "
                        "(runs_b1n -> results/b1n_compmeal_nn.json)")
    a = p.parse_args()
    out_json = RESULTS / ("b1n_compmeal_nn.json" if a.root == "runs_b1n" else f"{a.root}.json")
    per_user_root = RESULTS / a.root
    key = a.variant if (a.lr, a.wd) == (LR, WD) and a.root == "runs_b1n" else f"{a.variant}__lr{a.lr:g}_wd{a.wd:g}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = json.loads(out_json.read_text()) if out_json.exists() else {}
    for cell in a.cells:
        name, protocol = cell.split("/")
        t_start = time.time()
        base = load(name)
        uc = user_course_meta(name)
        cat = catalog(base)
        comps = completion_times(base, uc).drop_duplicates(["u", "m"])
        content = content_embedding(course_meta(name, columns=("course_name", "ingredients", "tags")), cat.used)
        sets, ev, data = simulate(protocol, base, uc, cat, comps)
        train, tune, test = (data.user_meal[s] for s in ("train", "tune", "test"))
        kw = {"use_prior": not protocol.startswith("cold"), "lr": a.lr, "weight_decay": a.wd, **VARIANTS[a.variant]}
        rows = []
        for seed in a.seeds:
            print(f"{cell} {key} seed {seed}", flush=True)
            r = train_compmeal_nn(sets, ev, tune, test, train, cat, base.course_category[cat.used], content,
                                  seed, device, epochs=a.epochs, **kw)
            users = np.flatnonzero(test.getnnz(axis=1))
            pu = per_user_metrics(topk(mask_scores(r["scores"][users], train[users]), 80), test[users])
            out = per_user_root / name / protocol / key
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"seed{seed}.npz", users=users, **pu)
            if a.save_scores:
                np.save(out / f"scores_seed{seed}.npy", r["scores"].astype(np.float16))
            rows.append({k: r[k] for k in ("epoch", "tune_curve", "tune", "test")} | {"seed": seed})
            print(f"{cell} {key} seed {seed}: epoch {r['epoch']}  tune R@20 {r['tune']['recall@20']:.4f}  "
                  f"test R@20 {r['test']['recall@20']:.4f}  N@20 {r['test']['ndcg@20']:.4f}", flush=True)
        rec = np.array([x["test"]["recall@20"] for x in rows])
        nd = np.array([x["test"]["ndcg@20"] for x in rows])
        result.setdefault(cell, {})[key] = {
            "test_recall@20_mean": float(rec.mean()), "test_recall@20_sd": float(rec.std(ddof=1)) if len(rec) > 1 else 0.0,
            "test_ndcg@20_mean": float(nd.mean()), "seeds": rows, "config": kw, "seconds": time.time() - t_start}
        print(f"{cell} {key}: test R@20 {rec.mean():.4f}±{result[cell][key]['test_recall@20_sd']:.4f}", flush=True)
        out_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
