"""G5 fairness check: re-tune CrossCBR and MultiCBR on P1 for Youshu, NetEase and iFashion.

The reference runs (results/runs/p1/) use each repo's best settings, chosen on the released
splits (for iFashion, the leaky one), while BundleComp picks its tree count on P1 tune. Here,
as D3b did for MealRec+, each model gets a small grid on P1 chosen by tune R@20 (seed 0):
lr {3e-4, 1e-3, 3e-3} x weight decay {default/10, default, default*10}; layers and every
other setting stay at the upstream values. The winner is rerun with 5 seeds (the default
setting reuses results/runs/p1/), then compared with BundleComp per user.

    python -m experiments.g5_tuning grid     # results/g5_grid_jobs.txt
    python scripts/run_queue.py --jobs-file results/g5_grid_jobs.txt
    python -m experiments.g5_tuning extend   # one step past the grid edge where the best sits on it
    python scripts/run_queue.py --jobs-file results/g5_extend_jobs.txt
    python -m experiments.g5_tuning select   # results/g5_final_jobs.txt
    python scripts/run_queue.py --jobs-file results/g5_final_jobs.txt
    python -m experiments.g5_tuning report   # results/g5_tuning.json
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
from scipy.stats import wilcoxon

from experiments.g3_compare import seed_mean
from experiments.train_bundle import UPSTREAM
from mealrec.data import BUNDLE_DATASETS
from mealrec.runs import RESULTS, per_user, summaries

MODELS = ("CrossCBR", "MultiCBR")
LRS = (3e-4, 1e-3, 3e-3)
SEEDS = (0, 1, 2, 3, 4)


def l2_grid(model: str, dataset: str) -> tuple[float, ...]:
    d = UPSTREAM[(model, dataset)]["l2_reg"]
    return (d / 10, d, d * 10)


def tag(lr: float, l2: float) -> str:
    return f"lr{lr:g}_l2{l2:g}"


def job(dataset, model, seed, lr, l2, root) -> str:
    return f"--protocol p1 --dataset {dataset} --model {model} --seed {seed} --lr {lr:g} --l2 {l2:g} --root {root} --tag {tag(lr, l2)}"


def grid() -> None:
    lines = [job(d, m, 0, lr, l2, "runs_tune")
             for d, m in itertools.product(BUNDLE_DATASETS, MODELS) for lr in LRS for l2 in l2_grid(m, d)]
    path = RESULTS / "g5_grid_jobs.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {path.relative_to(RESULTS.parent)}")


def best():
    df = summaries("p1", root="runs_tune").query("dataset in @BUNDLE_DATASETS")
    df = df[df["model"].str.split("__").str[0].isin(MODELS)]
    df["base"] = df["model"].str.split("__").str[0]
    return df, df.loc[df.groupby(["dataset", "base"])["tune_sel/tune_recall@20"].idxmax()]


def is_default(row) -> bool:
    up = UPSTREAM[(row.base, row.dataset)]
    return np.isclose(row.lr, up["lr"]) and np.isclose(row.l2, up["l2_reg"])


def extend() -> None:
    """Where the best setting is on the edge of the grid, add one step beyond it in that direction."""
    df, b = best()
    done = {(r.dataset, r.base, round(r.lr, 12), round(r.l2, 15)) for r in df.itertuples()}
    lines = []
    for r in b.itertuples():
        l2s = l2_grid(r.base, r.dataset)
        lr_opts = [r.lr] + ([r.lr / 3] if np.isclose(r.lr, min(LRS)) else [r.lr * 3] if np.isclose(r.lr, max(LRS)) else [])
        l2_opts = [r.l2] + ([r.l2 / 10] if np.isclose(r.l2, min(l2s)) else [r.l2 * 10] if np.isclose(r.l2, max(l2s)) else [])
        for lr, l2 in itertools.product(lr_opts, l2_opts):
            if (r.dataset, r.base, round(lr, 12), round(l2, 15)) not in done:
                lines.append(job(r.dataset, r.base, 0, lr, l2, "runs_tune"))
    path = RESULTS / "g5_extend_jobs.txt"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"{len(lines)} jobs -> {path.relative_to(RESULTS.parent)}")


def select() -> None:
    df, b = best()
    expected = len(BUNDLE_DATASETS) * len(MODELS) * len(LRS) * 3  # plus any extension runs
    if len(df) < expected:
        print(f"warning: only {len(df)}/{expected} grid runs finished", file=sys.stderr)
    lines = [job(r.dataset, r.base, s, r.lr, r.l2, "runs_tuned") for r in b.itertuples() if not is_default(r) for s in SEEDS]
    path = RESULTS / "g5_final_jobs.txt"
    path.write_text("\n".join(lines) + "\n")
    print(b[["dataset", "base", "lr", "l2", "tune_sel/tune_recall@20"]].to_string(index=False))
    print(f"{len(lines)} jobs -> {path.relative_to(RESULTS.parent)}")


def report() -> None:
    _, b = best()
    out = {}
    for r in b.itertuples():
        if is_default(r):
            root, model = "runs", r.base
        else:
            root, model = "runs_tuned", f"{r.base}__{tag(r.lr, r.l2)}"
        seeds = sorted(int(x.parent.name[4:]) for x in (RESULTS / root / "p1" / r.dataset / model).glob("seed*/summary.json"))
        pus = [per_user("p1", r.dataset, model, s, root=root) for s in seeds]
        default = [per_user("p1", r.dataset, r.base, s) for s in SEEDS]
        users, ours = seed_mean(sorted((RESULTS / "runs_g5" / r.dataset).glob("seed*.npz")))
        assert all((x["users"] == users).all() for x in pus + default)
        tuned_r = np.mean([x["recall@20"] for x in pus], axis=0)
        out.setdefault(r.dataset, {})[r.base] = {
            "setting": {"lr": r.lr, "l2": r.l2, "is_default": bool(is_default(r))}, "n_seeds": len(pus),
            "default_recall@20": float(np.mean([x["recall@20"].mean() for x in default])),
            "tuned_recall@20": float(tuned_r.mean()), "bundlecomp_recall@20": float(ours.mean()),
            "bundlecomp_minus_tuned": float((ours - tuned_r).mean()),
            "wilcoxon_p": float(wilcoxon(ours, tuned_r).pvalue)}
        o = out[r.dataset][r.base]
        print(f"{r.dataset:<9} {r.base:<9} default {o['default_recall@20']:.4f} -> tuned {o['tuned_recall@20']:.4f} "
              f"(lr {r.lr:g}, l2 {r.l2:g}{', default' if o['setting']['is_default'] else ''}) | BundleComp "
              f"{o['bundlecomp_recall@20']:.4f}  diff {o['bundlecomp_minus_tuned']:+.4f}  p {o['wilcoxon_p']:.2g}")
    (RESULTS / "g5_tuning.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    {"grid": grid, "extend": extend, "select": select, "report": report}[sys.argv[1]]()
