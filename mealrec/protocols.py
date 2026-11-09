"""Evaluation protocols. Each returns a MealRecPlus that downstream code uses unchanged.

orig  the released split: user_course.txt is unsplit, so it contains the courses of
      held-out (tune/test) meals (D1).
p1    mask: for every held-out pair (u, m), drop the user-course edges (u, c), c in m,
      unless (u, c) is also explained by one of u's training meals.
p2    temporal: a meal is "completed" when the user has reviewed its last course.
      Pairs completed up to a global cutoff train; later pairs are split by time into
      tune and test. Only user-course edges up to the cutoff are kept.

Cold-start (D8), as in CoHeat/DisCo: MEALS are split 7:1:2 (fixed seed) and every
interaction of a meal follows it, so tune/test meals never appear in training.
cold     the released, unsplit user_course.txt (leaks the courses of cold meals)
cold_p1  the same split with the P1 mask applied to user_course

Robustness variants: p2_q70 / p2_q90 move the P2 cutoff to the 70th / 90th percentile of
completion time; cold_s<k> / cold_p1_s<k> use meal-split seed k instead of 0.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mealrec.data import BUNDLE_DATASETS, SPLITS, MealRecPlus, load, load_bundle, to_matrix

PROTOCOLS = ("orig", "p1", "p2")
COLD_PROTOCOLS = ("cold", "cold_p1")
VARIANT_PROTOCOLS = ("p2_q70", "p2_q90", "cold_s1", "cold_p1_s1", "cold_s2", "cold_p1_s2")


def _binary(m: sp.spmatrix) -> sp.csr_matrix:
    m = sp.csr_matrix(m, dtype=np.float32)
    m.eliminate_zeros()
    m.data[:] = 1.0
    return m


def mask_heldout(data: MealRecPlus) -> MealRecPlus:
    mc = data.meal_course
    heldout = data.user_meal["tune"] + data.user_meal["test"]
    leaked = _binary(heldout @ mc)
    explained = _binary(data.user_meal["train"] @ mc)
    drop = data.user_course.multiply(leaked) - data.user_course.multiply(leaked).multiply(explained)
    return replace(data, name=f"{data.name}/p1", user_course=_binary(data.user_course - drop))


def completion_times(data: MealRecPlus, uc: pd.DataFrame) -> pd.DataFrame:
    """One row per (user, meal) positive with its split and completion time."""
    when = pd.Series(uc["t"].to_numpy(), index=pd.MultiIndex.from_arrays([uc["u"], uc["c"]]))
    when = when[~when.index.duplicated()]
    rows = []
    for split in SPLITS:
        users, meals = data.user_meal[split].nonzero()
        rows.append(pd.DataFrame({"u": users, "m": meals, "orig_split": split}))
    pairs = pd.concat(rows, ignore_index=True)
    mc = data.meal_course.tolil().rows
    courses = np.array([mc[m] for m in pairs["m"]])  # P x 3
    keys = pd.MultiIndex.from_arrays([np.repeat(pairs["u"].to_numpy(), 3), courses.ravel()])
    t = when.reindex(keys).to_numpy().reshape(-1, 3)
    pairs["t"] = pd.to_datetime(t.max(axis=1))
    return pairs


def temporal(data: MealRecPlus, train_share: float = 0.8, uc: pd.DataFrame | None = None) -> MealRecPlus:
    from mealrec.meta import user_course_meta

    if uc is None:
        uc = user_course_meta(data.name)
    pairs = completion_times(data, uc)
    cutoff = pairs["t"].quantile(train_share)
    later = pairs["t"] > cutoff
    split_at = pairs.loc[later, "t"].quantile(0.5)
    pairs["split"] = np.where(~later, "train", np.where(pairs["t"] <= split_at, "tune", "test"))

    shape = (data.n_users, data.n_meals)
    user_meal = {s: to_matrix(pairs.loc[pairs["split"] == s, ["u", "m"]].to_numpy(), shape) for s in SPLITS}
    kept = uc.loc[uc["t"] <= cutoff, ["u", "c"]].dropna().to_numpy(dtype=np.int64)
    # Keep only edges that exist in user_course.txt (the CSV is the superset it was built from).
    user_course = _binary(to_matrix(kept, (data.n_users, data.n_courses)).multiply(data.user_course))
    return replace(data, name=f"{data.name}/p2", user_course=user_course, user_meal=user_meal,
                   cutoffs={"train_until": str(cutoff), "tune_until": str(split_at)})


def cold_split(data: MealRecPlus, shares=(0.7, 0.1, 0.2), seed: int = 0) -> MealRecPlus:
    """Split meals (not pairs) into train/tune/test; each meal takes all its interactions."""
    meals = np.random.default_rng(seed).permutation(data.n_meals)
    cut = np.cumsum(np.round(np.array(shares[:2]) * data.n_meals).astype(int))
    which = np.empty(data.n_meals, dtype=np.int64)
    for i, part in enumerate(np.split(meals, cut)):
        which[part] = i
    allpos = _binary(sum(data.user_meal[s] for s in SPLITS))
    user_meal = {s: _binary(allpos @ sp.diags((which == i).astype(np.float32))) for i, s in enumerate(SPLITS)}
    return replace(data, name=f"{data.name}/cold", user_meal=user_meal)


def get(protocol: str, name: str) -> MealRecPlus:
    known = PROTOCOLS + COLD_PROTOCOLS + VARIANT_PROTOCOLS
    if protocol not in known:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {known}")
    if name in BUNDLE_DATASETS:  # no timestamps or categories: only the released split and its P1 mask
        if protocol not in ("orig", "p1"):
            raise ValueError(f"{name} supports protocols orig and p1 only")
        data = load_bundle(name)
        return mask_heldout(data) if protocol == "p1" else data
    data = load(name)
    if protocol.startswith("cold"):
        base, _, seed = protocol.partition("_s")
        split = cold_split(data, seed=int(seed or 0))
        if base == "cold_p1":
            split = mask_heldout(split)
        return replace(split, name=f"{name}/{protocol}")
    if protocol.startswith("p2_q"):
        return replace(temporal(data, train_share=int(protocol[4:]) / 100), name=f"{name}/{protocol}")
    if protocol == "p1":
        return mask_heldout(data)
    if protocol == "p2":
        return temporal(data)
    return data
