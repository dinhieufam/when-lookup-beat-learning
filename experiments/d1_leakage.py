"""D1: does user_course.txt leak the test meals?

Compares training-free baselines on the official MealRec+ split. CoverHeuristic
only looks up whether the user has interacted with each course of a meal.

    python -m experiments.d1_leakage
"""
from __future__ import annotations

import json
from pathlib import Path

from mealrec.baselines import BASELINES
from mealrec.data import DATASETS, load
from mealrec.diagnostics import leakage_stats
from mealrec.metrics import health_metrics, mask_scores, ranking_metrics, topk

OUT = Path(__file__).resolve().parent.parent / "results" / "d1_leakage.json"

# Best published numbers on the same split (Li et al., SIGIR 2024, Table 6: CrossCBR).
REPORTED_SOTA = {
    "MealRec+H": {"recall@20": 0.3753, "ndcg@20": 0.2588},
    "MealRec+L": {"recall@20": 0.1920, "ndcg@20": 0.1018},
}


def main() -> None:
    results = {}
    for name in DATASETS:
        data = load(name)
        test, train = data.user_meal["test"], data.user_meal["train"]
        users = test.getnnz(axis=1).nonzero()[0]
        row = {"leakage": leakage_stats(data), "reported_sota_crosscbr": REPORTED_SOTA[name], "baselines": {}}
        print(f"\n== {name} ==")
        lk = row["leakage"]
        print(f"P(all courses in user_course | test positive) = {lk['p_full_cover_given_positive']['test']:.3f}")
        print(f"P(positive | all courses in user_course)      = {lk['p_positive_given_full_cover']:.3f}")
        print(f"{'model':<16}{'R@20':>8}{'N@20':>8}{'R@80':>8}{'FSA':>8}{'ExpoGap':>9}")
        for model, fn in BASELINES.items():
            scores = fn(data)
            acc = ranking_metrics(scores, test, exclude=train)
            top = topk(mask_scores(scores[users], train[users]), 20)
            hl = health_metrics(top, users, data.meal_fsa, data.meal_who, data.user_fsa, data.user_who)
            row["baselines"][model] = {**acc, **{f"{k}@20": v for k, v in hl.items()}}
            print(f"{model:<16}{acc['recall@20']:>8.4f}{acc['ndcg@20']:>8.4f}{acc['recall@80']:>8.4f}"
                  f"{hl['fsa_mean']:>8.3f}{hl['expo_gap']:>9.4f}")
        s = REPORTED_SOTA[name]
        print(f"{'CrossCBR (paper)':<16}{s['recall@20']:>8.4f}{s['ndcg@20']:>8.4f}")
        results[name] = row
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
