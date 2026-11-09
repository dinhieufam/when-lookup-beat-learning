"""D9: does MealRec (2022, arXiv 2205.12133) share the MealRec+ leak?

MealRec ships meal.csv (meal -> appetizer, main, dessert), user_meal.csv and
user_recipe.csv with raw Allrecipes ids and no split. We map them onto MealRec+H
through meta_data/{user,course}2index.txt and test whether the two are the same data,
then measure the coverage leak on MealRec's own files.

Data: Google Drive folder linked from github.com/WUT-IDEA/MealRec, e.g.
    gdown --folder https://drive.google.com/drive/folders/1AhsODmR1xRJsY8VwOaQbfpi85GBw_0ZR -O data/raw/MealRec2022

    python -m experiments.d9_mealrec2022
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mealrec.data import DATA_ROOT, SPLITS, load
from mealrec.meta import read_index

OUT = Path(__file__).resolve().parent.parent / "results" / "d9_mealrec2022.json"
SRC = DATA_ROOT / "MealRec2022"


def main() -> None:
    meal = pd.read_csv(SRC / "meal.csv")
    user_meal = pd.read_csv(SRC / "user_meal.csv")
    user_recipe = pd.read_csv(SRC / "user_recipe.csv", usecols=["user_id", "recipe_id"])

    # Leak on MealRec's own files: share of user-meal pairs whose 3 recipes are all in user_recipe.
    touched = set(map(tuple, user_recipe.to_numpy()))
    courses = meal.set_index("bundle_id")[["appetizer", "main_dish", "dessert"]]
    triples = courses.loc[user_meal["meal_id"]].to_numpy()
    full = np.array([all((u, c) in touched for c in t) for u, t in zip(user_meal["user_id"], triples)])

    # Identity with MealRec+H, via raw ids.
    h = load("MealRec+H")
    meta = DATA_ROOT / "MealRec+H" / "meta_data"
    users = {v: k for k, v in read_index(meta / "user2index.txt").items()}
    recipes = {v: k for k, v in read_index(meta / "course2index.txt").items()}
    rows = h.meal_course.tolil().rows
    h_meal = {frozenset(recipes[c] for c in rows[m]): m for m in range(h.n_meals)}
    meal_map = {b: h_meal.get(frozenset(t)) for b, t in zip(courses.index, courses.to_numpy())}
    h_pairs = set()
    for s in SPLITS:
        uu, mm = h.user_meal[s].nonzero()
        h_pairs |= {(users[u], m) for u, m in zip(uu, mm)}
    pairs = {(u, meal_map[m]) for u, m in zip(user_meal["user_id"], user_meal["meal_id"])}
    uu, cc = h.user_course.nonzero()
    h_uc = {(users[u], recipes[c]) for u, c in zip(uu, cc)}

    result = {
        "mealrec2022": {
            "meals": int(len(meal)), "user_meal": int(len(user_meal)), "user_recipe": int(len(user_recipe)),
            "users": int(user_meal["user_id"].nunique()), "recipes": int(user_recipe["recipe_id"].nunique()),
            "p_full_cover_given_positive": float(full.mean()),
        },
        "identity_with_mealrec_plus_h": {
            "meals_matching_a_h_meal": int(sum(v is not None for v in meal_map.values())),
            "user_meal_pairs_in_h": int(len(pairs & h_pairs)), "h_user_meal_pairs": int(len(h_pairs)),
            "user_recipe_pairs_in_h_user_course": int(len(touched & h_uc)), "h_user_course": int(len(h_uc)),
        },
    }
    ident = result["identity_with_mealrec_plus_h"]
    result["same_data_as_mealrec_plus_h"] = bool(
        ident["meals_matching_a_h_meal"] == len(meal) == h.n_meals
        and ident["user_meal_pairs_in_h"] == len(pairs) == len(h_pairs)
        and ident["user_recipe_pairs_in_h_user_course"] == len(touched) == len(h_uc)
    )
    print(json.dumps(result, indent=2))
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
