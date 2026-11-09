"""D2: how much does choosing the epoch on the test set (as the MealRec+ paper does) inflate results?

Aggregates results/runs/orig/ (experiments/train_bundle.py, 5 seeds). For each model
and dataset: test R@20 / N@20 at the epoch chosen by tune vs by test, the relative
inflation, and whether the model ranking changes (Kendall tau).

    python scripts/run_queue.py --protocols orig --models LightGCN BGCN CrossCBR MultiCBR HyperMBR
    python -m experiments.d2_epoch_selection
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import kendalltau

from mealrec.runs import mean_sd, summaries

OUT = Path(__file__).resolve().parent.parent / "results" / "d2_epoch_selection.json"

# Li et al., SIGIR 2024, Table 6 (test-selected epoch), for reference.
REPORTED = {
    "MealRec+H": {"BGCN": (0.2083, 0.1263), "HyperMBR": (0.3144, 0.2153), "CrossCBR": (0.3753, 0.2588)},
    "MealRec+L": {"BGCN": (0.0882, 0.0438), "HyperMBR": (0.1497, 0.0786), "CrossCBR": (0.1920, 0.1018)},
}


def main() -> None:
    df = summaries("orig")
    if df.empty:
        raise SystemExit("no finished runs under results/runs/orig")
    cols = [f"{r}_sel/{m}" for r in ("tune", "test") for m in ("recall@20", "ndcg@20")]
    df["inflation/recall@20"] = df["test_sel/recall@20"] / df["tune_sel/recall@20"] - 1
    df["inflation/ndcg@20"] = df["test_sel/ndcg@20"] / df["tune_sel/ndcg@20"] - 1
    agg = mean_sd(df, cols + ["inflation/recall@20", "inflation/ndcg@20", "epochs_run", "sec_per_epoch"])

    result = {}
    for ds, g in agg.groupby("dataset"):
        rows = g.drop(columns="dataset").set_index("model").to_dict("index")
        for m, (r20, n20) in REPORTED.get(ds, {}).items():
            if m in rows:
                rows[m]["reported/recall@20"], rows[m]["reported/ndcg@20"] = r20, n20
        by_tune = g.set_index("model")["tune_sel/recall@20/mean"]
        by_test = g.set_index("model")["test_sel/recall@20/mean"]
        tau = kendalltau(by_tune.to_numpy(), by_test[by_tune.index].to_numpy()).statistic if len(g) > 1 else None
        result[ds] = {
            "models": rows,
            "ranking_by_tune_selection": by_tune.sort_values(ascending=False).index.tolist(),
            "ranking_by_test_selection": by_test.sort_values(ascending=False).index.tolist(),
            "kendall_tau": tau,
            "max_relative_inflation_recall@20": float(g["inflation/recall@20/mean"].max()),
        }
        print(f"\n== {ds} ==")
        print(f"{'model':<10}{'seeds':>6}{'R@20 tune-sel':>16}{'R@20 test-sel':>16}{'infl.':>8}{'reported':>10}")
        for m in result[ds]["ranking_by_tune_selection"]:
            r = rows[m]
            print(f"{m:<10}{r['n_seeds']:>6}{r['tune_sel/recall@20/mean']:>10.4f}±{r['tune_sel/recall@20/sd']:.4f}"
                  f"{r['test_sel/recall@20/mean']:>10.4f}±{r['test_sel/recall@20/sd']:.4f}"
                  f"{r['inflation/recall@20/mean']:>+8.1%}{r.get('reported/recall@20', float('nan')):>10.4f}")
        print(f"Kendall tau (tune vs test selection): {tau}")
    OUT.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nwrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
