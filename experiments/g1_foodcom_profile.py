"""G1: can Food.com give a third meal dataset with real (short) sessions?

Food.com (Majumder et al., EMNLP 2019; Kaggle shuyangli94/food-com-recipes-and-user-interactions)
has review dates and recipe tags. A recipe gets a category when exactly one of the tags
appetizers / main-dish / desserts applies. A session is a window of `w` days anchored at a
user's first positive review (rating >= 4) not yet in a session; a session forms meals when
it has all three categories. We count how many users and meals this gives and how often a
meal recurs across users (MealRec+ meals are shared by construction).

    python -m experiments.g1_foodcom_profile
"""
from __future__ import annotations

import ast
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "foodcom"
OUT = ROOT / "results" / "g1_foodcom_profile.json"
CATEGORY_TAGS = ("appetizers", "main-dish", "desserts")
WINDOWS = (1, 7, 30)


def positives() -> pd.DataFrame:
    recipes = pd.read_csv(RAW / "RAW_recipes.csv", usecols=["id", "tags"])
    tags = recipes["tags"].map(ast.literal_eval)
    hits = np.stack([tags.map(lambda t, c=c: c in t).to_numpy() for c in CATEGORY_TAGS], axis=1)
    recipes = recipes[hits.sum(axis=1) == 1].assign(cat=hits[hits.sum(axis=1) == 1].argmax(axis=1))
    inter = pd.read_csv(RAW / "RAW_interactions.csv", usecols=["user_id", "recipe_id", "date", "rating"],
                        parse_dates=["date"])
    x = inter[inter["rating"] >= 4].merge(recipes[["id", "cat"]], left_on="recipe_id", right_on="id")
    return x.assign(day=x["date"].to_numpy().astype("datetime64[D]").astype(np.int64))


def sessions(x: pd.DataFrame, w: int) -> pd.DataFrame:
    """One row per (user, session start, appetizer, main, dessert) triple inside a complete session."""
    rows = []
    for user, g in x.sort_values("day").groupby("user_id"):
        if g["cat"].nunique() < 3:
            continue
        d, c, rid = g["day"].to_numpy(), g["cat"].to_numpy(), g["recipe_id"].to_numpy()
        i = 0
        while i < len(d):
            j = i
            while j < len(d) and d[j] - d[i] < w:
                j += 1
            by_cat = defaultdict(list)
            for k in range(i, j):
                by_cat[c[k]].append(rid[k])
            if len(by_cat) == 3:
                rows.extend((user, d[i], *t) for t in itertools.product(by_cat[0], by_cat[1], by_cat[2]))
            i = j
    return pd.DataFrame(rows, columns=["u", "start", "a", "m", "d"])


def main() -> None:
    x = positives()
    out = {
        "positive_reviews_on_categorised_recipes": int(len(x)),
        "users": int(x["user_id"].nunique()),
        "by_category": {CATEGORY_TAGS[k]: int(v) for k, v in x["cat"].value_counts().sort_index().items()},
        "date_range": [str(x["date"].min().date()), str(x["date"].max().date())],
        "windows": {},
    }
    for w in WINDOWS:
        s = sessions(x, w)
        per_meal = s.groupby(["a", "m", "d"])["u"].nunique()
        row = {
            "sessions": int(s.groupby(["u", "start"]).ngroups),
            "users": int(s["u"].nunique()),
            "user_meal_pairs": int(len(s.drop_duplicates(["u", "a", "m", "d"]))),
            "distinct_meals": int(len(per_meal)),
            "meals_with_2plus_users": int((per_meal >= 2).sum()),
            "courses": int(pd.unique(s[["a", "m", "d"]].to_numpy().ravel()).size),
            "courses_with_2plus_users": {col: int((s.groupby(col)["u"].nunique() >= 2).sum()) for col in "amd"},
        }
        out["windows"][f"{w}d"] = row
        print(f"w={w}d: {row}")
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
