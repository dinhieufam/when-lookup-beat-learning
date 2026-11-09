"""D8: is the cold-start advantage on MealRec+ real, or the user_course leak again?

DisCo (WWW'25 Companion) reports, on "Meal" in the cold-bundle setting, R@20 of
0.0022 (CrossCBR), 0.0102 (MultiCBR), 0.1650 (CoHeat) and 0.2838 (DisCo): models
that represent a bundle through its items win by far. In MealRec+ a cold meal's
courses are already in user_course.txt for exactly the users who will take it.

Protocols (mealrec.protocols): `cold` splits meals 7:1:2 and keeps the released
user_course; `cold_p1` also masks the user-course edges that only held-out meals
explain. Models: CoHeat (snudatalab/CoHeat, 1 layer, 100 epochs, no early stopping,
as upstream), CrossCBR and MultiCBR (settings as in D2), 5 seeds; training-free
baselines as references (CoverHeuristic is the leak canary, TrainCover the
legitimate ceiling).

    python -m experiments.d8_cold_start jobs     # writes results/d8_jobs.txt
    python scripts/run_queue.py --jobs-file results/d8_jobs.txt
    python -m experiments.d8_cold_start report   # writes results/d8_cold_start.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

from mealrec.baselines import BASELINES
from mealrec.data import DATASETS
from mealrec.metrics import ranking_metrics
from mealrec.protocols import COLD_PROTOCOLS, get
from mealrec.runs import mean_sd, summaries

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEEDS = (0, 1, 2, 3, 4)
MODELS = {
    "CoHeat": "--layers 1 --min-epochs 100",
    "CrossCBR": "",
    "MultiCBR": "",
}
# DisCo, arXiv 2505.14901, cold-bundle "Meal" (H or L not stated), R@20 / N@20.
REPORTED = {"CrossCBR": (0.0022, 0.0005), "MultiCBR": (0.0102, 0.0054), "CoHeat": (0.1650, 0.1084),
            "DisCo": (0.2838, 0.1961)}


def jobs() -> None:
    lines = [f"--protocol {p} --dataset {d} --model {m} --seed {s} {extra}".strip()
             for (m, extra), p, d, s in itertools.product(MODELS.items(), COLD_PROTOCOLS, DATASETS, SEEDS)]
    path = RESULTS / "d8_jobs.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {path.relative_to(RESULTS.parent)}")


def report() -> None:
    runs = summaries()
    runs = runs[runs["protocol"].isin(COLD_PROTOCOLS)]
    cols = ["tune_sel/recall@20", "tune_sel/ndcg@20", "test_sel/recall@20"]
    agg = mean_sd(runs, cols, by=("protocol", "dataset", "model")) if not runs.empty else None
    result = {"reported_disco_meal": REPORTED}
    for dataset, protocol in itertools.product(DATASETS, COLD_PROTOCOLS):
        data = get(protocol, dataset)
        train, test = data.user_meal["train"], data.user_meal["test"]
        base = {b: ranking_metrics(fn(data), test, exclude=train) for b, fn in BASELINES.items()}
        models = {}
        if agg is not None:
            g = agg[(agg.protocol == protocol) & (agg.dataset == dataset)]
            models = g.drop(columns=["protocol", "dataset"]).set_index("model").to_dict("index")
        result.setdefault(dataset, {})[protocol] = {
            "baselines": base, "models": models,
            "cold_meals": {s: int((data.user_meal[s].getnnz(axis=0) > 0).sum()) for s in data.user_meal},
            "user_course_edges": int(data.user_course.nnz),
        }
        print(f"\n== {dataset} {protocol}")
        for b, m in base.items():
            print(f"  {b:<15} R@20 {m['recall@20']:.4f}  N@20 {m['ndcg@20']:.4f}")
        for m, r in sorted(models.items(), key=lambda kv: -kv[1]["tune_sel/recall@20/mean"]):
            print(f"  {m:<15} R@20 {r['tune_sel/recall@20/mean']:.4f}±{r['tune_sel/recall@20/sd']:.4f}  "
                  f"N@20 {r['tune_sel/ndcg@20/mean']:.4f}  (test-selected R@20 {r['test_sel/recall@20/mean']:.4f}, "
                  f"n={r['n_seeds']})")
    out = RESULTS / "d8_cold_start.json"
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nwrote {out.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    {"jobs": jobs, "report": report}[sys.argv[1]]()
