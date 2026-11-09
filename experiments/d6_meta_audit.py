"""D6 (first pass): audit what the metadata says about sessions, categories and FSA.

1. Session span: days between the reviews of a meal's three courses by the same user.
   A real dining session should be ~0 days.
2. Course usage: how many courses appear in any meal.
3. Category labels: share of dessert-like names per category (rough noise signal).
4. FSA: can course_fsa.txt be reproduced from per-serving nutrition with the
   UK FSA front-of-pack thresholds (per 100 g)? Low agreement means the scoring
   recipe is not recoverable from the released data.

Needs `bash scripts/download_mealrecplus.sh --meta`.

    python -m experiments.d6_meta_audit
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mealrec.data import DATASETS, SPLITS, load
from mealrec.meta import course_meta, nutrient, user_course_meta

OUT = Path(__file__).resolve().parent.parent / "results" / "d6_meta_audit.json"
DESSERT_WORDS = r"cake|cookie|pie\b|brownie|pudding|ice cream|fudge|cheesecake|tart\b|cobbler|candy|frosting|cupcake"


def traffic_light(x: np.ndarray, green: float, red: float) -> np.ndarray:
    return np.where(x <= green, 1, np.where(x > red, 3, 2))


def session_spans(data, uc) -> dict:
    when = {(u, c): t for u, c, t in zip(uc["u"], uc["c"], uc["t"])}
    mc = data.meal_course.tolil().rows
    out = {}
    for split in SPLITS:
        users, meals = data.user_meal[split].nonzero()
        days = np.array([
            (max(ts) - min(ts)).days
            for ts in ([when[(u, c)] for c in mc[m]] for u, m in zip(users, meals))
        ])
        out[split] = {
            "median_days": float(np.median(days)),
            "p25_days": float(np.percentile(days, 25)),
            "p75_days": float(np.percentile(days, 75)),
            "share_same_day": float(np.mean(days == 0)),
            "share_over_1_year": float(np.mean(days > 365)),
        }
    return out


def main() -> None:
    results = {}
    for name in DATASETS:
        data = load(name)
        uc = user_course_meta(name)
        meta = course_meta(name, columns=("course_name", "category", "nutritions"))
        used = np.flatnonzero(np.asarray(data.meal_course.sum(axis=0)).ravel() > 0)

        row = {
            "ratings": {int(k): int(v) for k, v in uc["rating"].value_counts().sort_index().items()},
            "review_dates": [str(uc["t"].min().date()), str(uc["t"].max().date())],
            "session_span": session_spans(data, uc),
            "courses_in_any_meal": int(len(used)),
            "courses_total": int(data.n_courses),
        }
        used_meta = meta.loc[used]
        row["dessert_word_share_by_category"] = {
            cat: float(g["course_name"].str.contains(DESSERT_WORDS, case=False).mean())
            for cat, g in used_meta.groupby("category")
        }

        n = meta["nutritions"]
        fat, sat, sugar, sodium = (nutrient(n, k) for k in ("fat", "saturatedFat", "sugars", "sodium"))
        salt = sodium * 2.5 / 1000
        fsa_100g = (traffic_light(fat, 3, 17.5) + traffic_light(sat, 1.5, 5)
                    + traffic_light(sugar, 5, 22.5) + traffic_light(salt, 0.3, 1.5))
        released = data.course_fsa[meta.index.to_numpy()]
        row["fsa_recompute_agreement_per_serving_vs_100g_thresholds"] = float(np.mean(fsa_100g == released))
        row["meal_fsa"] = {"mean": float(data.meal_fsa.mean()), "sd": float(data.meal_fsa.std()),
                           "share_le_8": float(np.mean(data.meal_fsa <= 8))}
        row["meal_who"] = {"mean": float(data.meal_who.mean()), "sd": float(data.meal_who.std())}

        results[name] = row
        print(f"\n== {name} ==")
        print(json.dumps(row, indent=2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
