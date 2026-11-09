"""D3b: are the clean-protocol conclusions robust to tuning the models on MealRec+?

D2/D3 used each repo's best settings for other datasets (Youshu/NetEase). Here every
model gets a small grid on P1 and P2, chosen by tune R@20 (seed 0); the winning
setting is then rerun with 5 seeds and compared with the default settings and with
the training-free baselines.

    python -m experiments.d3b_tuning grid      # writes results/d3b_grid_jobs.txt
    python scripts/run_queue.py --jobs-file results/d3b_grid_jobs.txt
    python -m experiments.d3b_tuning extend    # one step past the grid edge where the best sits on it
    python scripts/run_queue.py --jobs-file results/d3b_extend_jobs.txt
    python -m experiments.d3b_tuning select    # writes results/d3b_final_jobs.txt
    python scripts/run_queue.py --jobs-file results/d3b_final_jobs.txt
    python -m experiments.d3b_tuning report    # writes results/d3b_tuning.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

from mealrec.baselines import BASELINES
from mealrec.data import DATASETS
from mealrec.metrics import ranking_metrics
from mealrec.protocols import get
from mealrec.runs import mean_sd, summaries

RESULTS = Path(__file__).resolve().parent.parent / "results"
PROTOCOLS = ("p1", "p2")
LRS = (3e-4, 1e-3, 3e-3)
# L2 means weight decay for CrossCBR/MultiCBR, a per-batch embedding penalty for LightGCN
# and BGCN's embedding "decay"; the grids bracket each repo's default.
GRID = {
    "LightGCN": {"lr": LRS, "l2": (1e-5, 1e-4, 1e-3), "layers": (1, 2)},
    "BGCN": {"lr": LRS, "l2": (1e-7, 1e-5, 1e-3), "layers": (1, 2)},
    "CrossCBR": {"lr": LRS, "l2": (1e-6, 1e-5, 1e-4), "layers": (1, 2)},
    "MultiCBR": {"lr": LRS, "l2": (1e-6, 1e-5, 1e-4), "layers": (2,)},  # fusion weights are set for 2 layers
}
SEEDS = (0, 1, 2, 3, 4)


def tag(lr, l2, layers) -> str:
    return f"lr{lr:g}_l2{l2:g}_L{layers}"


def job(protocol, dataset, model, seed, lr, l2, layers, root) -> str:
    return (f"--protocol {protocol} --dataset {dataset} --model {model} --seed {seed} "
            f"--lr {lr:g} --l2 {l2:g} --layers {layers} --root {root} --tag {tag(lr, l2, layers)}")


def grid() -> None:
    lines = []
    for protocol, dataset, (model, g) in itertools.product(PROTOCOLS, DATASETS, GRID.items()):
        for lr, l2, layers in itertools.product(g["lr"], g["l2"], g["layers"]):
            lines.append(job(protocol, dataset, model, 0, lr, l2, layers, "runs_tune"))
    path = RESULTS / "d3b_grid_jobs.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {path.relative_to(RESULTS.parent)}")


def best_settings():
    df = summaries(root="runs_tune")
    best = df.loc[df.groupby(["protocol", "dataset", "model"])["tune_sel/tune_recall@20"].idxmax()]
    return df, best


def select() -> None:
    df, best = best_settings()
    expected = sum(len(list(itertools.product(*g.values()))) for g in GRID.values()) * len(PROTOCOLS) * len(DATASETS)
    if len(df) < expected:
        print(f"warning: only {len(df)}/{expected} grid runs finished", file=sys.stderr)
    lines = [job(r.protocol, r.dataset, r.model, s, r.lr, r.l2, r.layers, "runs_tuned")
             for r in best.itertuples() for s in SEEDS]
    path = RESULTS / "d3b_final_jobs.txt"
    path.write_text("\n".join(lines) + "\n")
    print(best[["protocol", "dataset", "model", "tag", "tune_sel/tune_recall@20"]].to_string(index=False))
    print(f"{len(lines)} jobs -> {path.relative_to(RESULTS.parent)}")


def extend() -> None:
    """One more step beyond the top edge of the grid wherever the best setting sits on it."""
    _, best = best_settings()
    lines = []
    for r in best.itertuples():
        g = GRID[r.model]
        lrs = [r.lr] + ([r.lr * 10 / 3] if r.lr == max(g["lr"]) else [])  # 3e-3 -> 1e-2
        l2s = [r.l2] + ([r.l2 * 10] if r.l2 == max(g["l2"]) else [])
        for lr, l2 in itertools.product(lrs, l2s):
            if (lr, l2) != (r.lr, r.l2):
                lines.append(job(r.protocol, r.dataset, r.model, 0, round(lr, 6), l2, r.layers, "runs_tune"))
    path = RESULTS / "d3b_extend_jobs.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {path.relative_to(RESULTS.parent)}")


def report() -> None:
    grid_df, best = best_settings()
    # Only the selected setting of each cell (an earlier selection may have left other settings behind).
    finals = summaries(root="runs_tuned").merge(best[["protocol", "dataset", "model", "tag"]])
    tuned = mean_sd(finals, ["tune_sel/recall@20", "tune_sel/ndcg@20"], by=("protocol", "dataset", "model"))
    default = mean_sd(summaries(), ["tune_sel/recall@20"], by=("protocol", "dataset", "model"))
    result = {}
    for protocol, dataset in itertools.product(PROTOCOLS, DATASETS):
        data = get(protocol, dataset)
        train, test = data.user_meal["train"], data.user_meal["test"]
        base = {b: ranking_metrics(fn(data), test, exclude=train)["recall@20"] for b, fn in BASELINES.items()}
        rows = {}
        for model in GRID:
            sel = lambda d: d[(d.protocol == protocol) & (d.dataset == dataset) & (d.model == model)]
            t, dflt, b = sel(tuned), sel(default), sel(best)
            if t.empty:
                continue
            rows[model] = {
                "setting": b["tag"].iloc[0],
                "tuned/recall@20/mean": float(t["tune_sel/recall@20/mean"].iloc[0]),
                "tuned/recall@20/sd": float(t["tune_sel/recall@20/sd"].iloc[0]),
                "tuned/ndcg@20/mean": float(t["tune_sel/ndcg@20/mean"].iloc[0]),
                "n_seeds": int(t["n_seeds"].iloc[0]),
                "default/recall@20/mean": float(dflt["tune_sel/recall@20/mean"].iloc[0]) if not dflt.empty else None,
                "grid_spread_test_recall@20": [float(x) for x in
                                               (lambda g: (g.min(), g.max()))(sel(grid_df)["tune_sel/recall@20"])],
            }
        best_model = max(rows, key=lambda m: rows[m]["tuned/recall@20/mean"]) if rows else None
        legit = {k: v for k, v in base.items() if k != "CoverHeuristic" or protocol == "p2"}
        best_base = max(legit, key=legit.get)
        result.setdefault(dataset, {})[protocol] = {
            "models": rows, "baselines": base, "best_model": best_model,
            "best_legit_baseline": best_base,
            "best_model_beats_best_legit_baseline": bool(best_model and rows[best_model]["tuned/recall@20/mean"] > legit[best_base]),
        }
        print(f"\n== {dataset} {protocol}  (best legitimate baseline: {best_base} {legit[best_base]:.4f})")
        for m, r in sorted(rows.items(), key=lambda kv: -kv[1]["tuned/recall@20/mean"]):
            d = r["default/recall@20/mean"]
            print(f"  {m:<9} tuned {r['tuned/recall@20/mean']:.4f}±{r['tuned/recall@20/sd']:.4f} "
                  f"(default {d:.4f}) [{r['setting']}]")
    out = RESULTS / "d3b_tuning.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    {"grid": grid, "select": select, "extend": extend, "report": report}[sys.argv[1]]()
