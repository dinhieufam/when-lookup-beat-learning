"""Statistics that show how much user_course.txt reveals about held-out meals."""
from __future__ import annotations

import numpy as np

from mealrec.baselines import course_coverage
from mealrec.data import SPLITS, MealRecPlus


def leakage_stats(data: MealRecPlus) -> dict:
    sizes = np.asarray(data.meal_course.sum(axis=1)).ravel()
    full = course_coverage(data) >= sizes[None, :]
    positives = sum(data.user_meal[s] for s in SPLITS).toarray() > 0
    train = data.user_meal["train"].toarray() > 0
    test = data.user_meal["test"].toarray() > 0
    tune = data.user_meal["tune"].toarray() > 0

    out = {
        "meal_sizes": sorted(set(sizes.astype(int).tolist())),
        "pairs": {s: int(data.user_meal[s].nnz) for s in SPLITS},
        # Share of positive (user, meal) pairs whose courses all appear in user_course.txt.
        "p_full_cover_given_positive": {
            s: float(full[data.user_meal[s].toarray() > 0].mean()) for s in SPLITS
        },
        "p_full_cover_random_pair": float(full.mean()),
        # How deterministic the label is: share of fully covered pairs that are positive.
        "p_positive_given_full_cover": float(positives[full].mean()),
    }

    test_users = np.flatnonzero(test.any(axis=1))
    cand = full & ~train
    out["test_users"] = int(len(test_users))
    out["candidates_per_test_user"] = {
        "median": float(np.median(cand[test_users].sum(axis=1))),
        "mean": float(cand[test_users].sum(axis=1).mean()),
        "of_which_test": float(test[test_users].sum(axis=1).mean()),
        "of_which_tune": float((cand & tune)[test_users].sum(axis=1).mean()),
    }

    # Course edges that only a test meal accounts for (would vanish under a clean split).
    mc = data.meal_course
    test_courses = (data.user_meal["test"] @ mc).toarray() > 0
    other_courses = ((data.user_meal["train"] + data.user_meal["tune"]) @ mc).toarray() > 0
    out["user_course_edges"] = int(data.user_course.nnz)
    out["edges_only_from_test_meals"] = int((test_courses & ~other_courses).sum())
    out["courses_in_any_meal"] = int((mc.getnnz(axis=0) > 0).sum())
    return out
