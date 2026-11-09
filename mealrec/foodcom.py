"""Food.com as an open-composition meal benchmark with real, short sessions (G1, G4a).

Data: Majumder et al. (EMNLP 2019), Kaggle shuyangli94/food-com-recipes-and-user-interactions,
in data/raw/foodcom/ (see data/README.md). A recipe gets a slot when exactly one of the tags
appetizers / main-dish / desserts applies. A session is a window of `window` days anchored at
a user's first positive review (rating >= 4) of a slotted recipe not yet in a session; it
counts when it covers all three slots. Each session holds a set of recipes per slot.

Sessions are split by start time, like P2: up to the `train_share` quantile train, the rest
halved by time into tune and test. History for a session = all of the user's reviews (any
recipe, any rating) strictly before the session start.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "foodcom"
CATEGORY_TAGS = ("appetizers", "main-dish", "desserts")
DAY = np.timedelta64(1, "D")


@dataclass
class FoodCom:
    reviews: pd.DataFrame     # user_id, recipe_id, date (datetime64[D]), rating; all reviews
    slot: pd.Series           # recipe_id -> 0 appetizer / 1 main / 2 dessert (slotted recipes only)
    recipes: pd.DataFrame     # RAW_recipes indexed by id (name, tags, ingredients, nutrition, ...)
    sessions: pd.DataFrame    # session, user_id, start, split, slot, recipe_id (one row per recipe)
    cutoffs: dict


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    recipes = pd.read_csv(RAW / "RAW_recipes.csv").set_index("id")
    reviews = pd.read_csv(RAW / "RAW_interactions.csv", usecols=["user_id", "recipe_id", "date", "rating"],
                          parse_dates=["date"])
    reviews["date"] = reviews["date"].to_numpy().astype("datetime64[D]")
    return reviews, recipes


def slots(recipes: pd.DataFrame) -> pd.Series:
    tags = recipes["tags"].map(ast.literal_eval)
    hits = np.stack([tags.map(lambda t, c=c: c in t).to_numpy() for c in CATEGORY_TAGS], axis=1)
    one = hits.sum(axis=1) == 1
    return pd.Series(hits[one].argmax(axis=1), index=recipes.index[one], name="slot")


def build_sessions(reviews: pd.DataFrame, slot: pd.Series, window: int) -> pd.DataFrame:
    x = reviews[(reviews["rating"] >= 4) & reviews["recipe_id"].isin(slot.index)].copy()
    x["slot"] = slot.reindex(x["recipe_id"]).to_numpy()
    x = x.sort_values(["user_id", "date", "recipe_id"])
    rows, sid = [], 0
    for user, g in x.groupby("user_id", sort=False):
        if g["slot"].nunique() < 3:
            continue
        d = g["date"].to_numpy()
        i = 0
        while i < len(d):
            j = np.searchsorted(d, d[i] + window * DAY, side="left")
            part = g.iloc[i:j]
            if part["slot"].nunique() == 3:
                rows.append(part.assign(session=sid, start=d[i])[["session", "user_id", "start", "slot", "recipe_id"]])
                sid += 1
            i = j
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["session", "recipe_id"])
    return out


def load(window: int = 1, train_share: float = 0.8) -> FoodCom:
    reviews, recipes = load_raw()
    slot = slots(recipes)
    s = build_sessions(reviews, slot, window)
    starts = s.groupby("session")["start"].first()
    cut = starts.quantile(train_share)
    later = starts[starts > cut]
    split_at = later.quantile(0.5)
    split = pd.Series(np.where(starts <= cut, "train", np.where(starts <= split_at, "tune", "test")), index=starts.index)
    s["split"] = split.reindex(s["session"]).to_numpy()
    return FoodCom(reviews, slot, recipes, s, {"train_until": str(cut), "tune_until": str(split_at)})
