"""D3: who wins once the leak is removed?

1. Canary: training-free baselines on every protocol (mealrec.protocols). CoverHeuristic
   reads user_course; TrainCover only reads courses of the user's training meals, which
   every model is given, so it is the legitimate ceiling a clean canary should not exceed.
2. Leaderboard: trained models (results/runs/<protocol>/, epoch selected on tune, 5 seeds)
   on orig, p1 and p2; Kendall tau of each clean leaderboard against the original one;
   whether "cooperative (course-aware) > user-meal only" still holds.

    python scripts/run_queue.py --protocols orig p1 p2 --models LightGCN BGCN CrossCBR MultiCBR HyperMBR
    python -m experiments.d3_clean_protocols
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import kendalltau

from mealrec.baselines import BASELINES
from mealrec.data import DATASETS
from mealrec.metrics import ranking_metrics
from mealrec.protocols import PROTOCOLS, get
from mealrec.runs import COURSE_AWARE, mean_sd, summaries

OUT = Path(__file__).resolve().parent.parent / "results" / "d3_clean_protocols.json"


def canary() -> dict:
    out = {}
    for name in DATASETS:
        for protocol in PROTOCOLS:
            data = get(protocol, name)
            train, test = data.user_meal["train"], data.user_meal["test"]
            row = {b: ranking_metrics(fn(data), test, exclude=train) for b, fn in BASELINES.items()}
            row["pairs"] = {s: int(m.nnz) for s, m in data.user_meal.items()}
            row["user_course_edges"] = int(data.user_course.nnz)
            row["cutoffs"] = getattr(data, "cutoffs", None)
            leak = row["CoverHeuristic"]["recall@20"] - row["TrainCover"]["recall@20"]
            row["canary_minus_legit_ceiling_recall@20"] = leak
            out.setdefault(name, {})[protocol] = row
            print(f"{name:<10} {protocol:<5} " + "  ".join(
                f"{b} {row[b]['recall@20']:.4f}" for b in BASELINES) + f"  | canary - ceiling {leak:+.4f}")
    return out


def leaderboard() -> dict:
    df = summaries()
    if df.empty:
        return {}
    cols = ["tune_sel/recall@20", "tune_sel/ndcg@20", "tune_sel/recall@80"]
    agg = mean_sd(df, cols, by=("protocol", "dataset", "model"))
    out = {}
    for (protocol, ds), g in agg.groupby(["protocol", "dataset"]):
        out.setdefault(ds, {})[protocol] = {
            "models": g.drop(columns=["protocol", "dataset"]).set_index("model").to_dict("index"),
            "ranking": g.sort_values("tune_sel/recall@20/mean", ascending=False)["model"].tolist(),
        }
    for ds, by_protocol in out.items():
        ref = by_protocol.get("orig")
        for protocol, row in by_protocol.items():
            r = {m: v["tune_sel/recall@20/mean"] for m, v in row["models"].items()}
            if ref and protocol != "orig":
                common = [m for m in ref["models"] if m in r]
                if len(common) > 1:
                    row["kendall_tau_vs_orig"] = kendalltau(
                        [ref["models"][m]["tune_sel/recall@20/mean"] for m in common],
                        [r[m] for m in common]).statistic
            if "LightGCN" in r:
                row["course_aware_minus_lightgcn_recall@20"] = {m: r[m] - r["LightGCN"] for m in COURSE_AWARE if m in r}
            print(f"\n{ds} {protocol}: " + "  ".join(
                f"{m} {row['models'][m]['tune_sel/recall@20/mean']:.4f}±{row['models'][m]['tune_sel/recall@20/sd']:.4f}"
                f"(n={row['models'][m]['n_seeds']})" for m in row["ranking"]))
            if "kendall_tau_vs_orig" in row:
                print(f"  Kendall tau vs orig: {row['kendall_tau_vs_orig']:.3f}")
    return out


def main() -> None:
    result = {"canary": canary(), "leaderboard": leaderboard()}
    OUT.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nwrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
