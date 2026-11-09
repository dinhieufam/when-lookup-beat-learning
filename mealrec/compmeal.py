"""CompMeal (direction B1): a compositional meal scorer built from per-course signals.

A meal is one appetizer, one main and one dessert, so it is scored from what the user
did with each of its courses (reviewed it? how recently? how similar is it to what they
review?) plus meal-level signals (popularity, how often its course pairs co-occur).
A learning-to-rank model (experiments/b1_compmeal.py) combines them.

Every feature is computed "as of" a time t from reviews and meal completions up to t
only, so training windows can mimic the temporal protocol P2: features at t0, labels =
meals completed in (t0, t1].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mealrec.data import MealRecPlus

DAY = np.timedelta64(1, "D")
RECENT = 365  # days


@dataclass
class Catalog:
    """Static meal structure in a course space (by default: the courses that appear in some meal)."""
    used: np.ndarray        # course ids of the course space
    triples: np.ndarray     # M x 3, used-space index of the appetizer, main and dessert of each meal
    meal_course: sp.csr_matrix  # M x Cu
    pair_counts: np.ndarray     # M x 3, OTHER counted meals sharing (app, main), (app, dessert), (main, dessert)


PAIRS = ((0, 1), (0, 2), (1, 2))


def catalog(data: MealRecPlus, used: np.ndarray | None = None, pair_meals: np.ndarray | None = None) -> Catalog:
    """`used`: course space (default: courses in some meal). `pair_meals`: boolean mask of the meals
    whose course pairs are counted (default: all); a meal never counts itself."""
    if used is None:
        used = np.flatnonzero(data.meal_course.getnnz(axis=0))
    to_used = np.full(data.n_courses, -1)
    to_used[used] = np.arange(len(used))
    rows = data.meal_course.tolil().rows
    triples = np.zeros((data.n_meals, 3), dtype=np.int64)
    for m, courses in enumerate(rows):
        for c in courses:
            triples[m, data.course_category[c]] = to_used[c]
    mc = data.meal_course[:, used].tocsr()
    counted = np.ones(data.n_meals, bool) if pair_meals is None else np.asarray(pair_meals, bool)
    counts = np.zeros((data.n_meals, 3))
    for j, (a, b) in enumerate(PAIRS):
        key = triples[:, a] * len(used) + triples[:, b]
        uniq, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv, weights=counted, minlength=len(uniq))
        counts[:, j] = cnt[inv] - counted
    return Catalog(used, triples, mc, counts)


def pair_count_matrix(cat: Catalog, j: int, meals: np.ndarray) -> sp.csr_matrix:
    """Cu x Cu counts of the given meals (boolean mask) containing each course pair of slot pair j."""
    a, b = PAIRS[j]
    t = cat.triples[np.asarray(meals, bool)]
    n = len(cat.used)
    m = sp.csr_matrix((np.ones(len(t)), (t[:, a], t[:, b])), shape=(n, n))
    m.sum_duplicates()
    return m


def reviews_table(data: MealRecPlus, uc: pd.DataFrame) -> pd.DataFrame:
    """One row per user-course edge of user_course.txt, with the time of its (first) review."""
    uc = uc.dropna(subset=["u", "c"]).astype({"u": np.int64, "c": np.int64})
    uc = uc.sort_values("t").drop_duplicates(["u", "c"])
    u, c = data.user_course.nonzero()
    keep = pd.MultiIndex.from_arrays([u, c], names=["u", "c"])
    return uc.set_index(["u", "c"]).reindex(keep).dropna(subset=["t"]).reset_index()[["u", "c", "t"]]


@dataclass
class State:
    """What is visible as of a time t, at course level (U x Cu over used courses) and above."""
    reviewed: np.ndarray        # U x Cu bool
    log_days_since: np.ndarray  # U x Cu, NaN where not reviewed
    cf: np.ndarray              # U x Cu, mean cosine to the user's reviewed courses
    in_known: np.ndarray        # U x Cu bool, course is in a meal the user has completed
    log_pop: np.ndarray         # Cu
    log_pop_recent: np.ndarray  # Cu
    meal_log_pop: np.ndarray    # M
    meal_log_pop_recent: np.ndarray  # M
    user: dict[str, np.ndarray]  # name -> U
    known: sp.csr_matrix         # U x M, meals completed by t


def state_at(t: np.datetime64, data: MealRecPlus, cat: Catalog, reviews: pd.DataFrame,
             completions: pd.DataFrame) -> State:
    nu, nm, ncu = data.n_users, data.n_meals, len(cat.used)
    to_used = np.full(data.n_courses, -1)
    to_used[cat.used] = np.arange(ncu)

    r = reviews[reviews["t"] <= t]
    age = ((t - r["t"].to_numpy()) / DAY).astype(np.float32)
    recent = age <= RECENT
    cu = to_used[r["c"].to_numpy()]
    inmeal = cu >= 0
    uu = r["u"].to_numpy()

    # User x used-course: reviewed, days since review.
    R = sp.csr_matrix((np.ones(inmeal.sum(), np.float32), (uu[inmeal], cu[inmeal])), shape=(nu, ncu))
    days = np.full((nu, ncu), np.nan, np.float32)
    days[uu[inmeal], cu[inmeal]] = age[inmeal]

    # Item-item cosine between used courses, from co-reviews up to t.
    co = (R.T @ R).toarray()
    deg = np.sqrt(np.diag(co)).clip(min=1)
    cos = co / deg[:, None] / deg[None, :]
    np.fill_diagonal(cos, 0)
    n_rev = np.asarray(R.sum(axis=1)).ravel()
    cf = np.asarray(R @ cos) / n_rev.clip(min=1)[:, None]

    pop = np.asarray(R.sum(axis=0)).ravel()
    pop_recent = np.bincount(cu[inmeal & recent], minlength=ncu)

    # Meals completed (known) by t.
    k = completions[completions["t"] <= t]
    K = sp.csr_matrix((np.ones(len(k), np.float32), (k["u"].to_numpy(), k["m"].to_numpy())), shape=(nu, nm))
    K.data[:] = 1
    k_recent = k[((t - k["t"].to_numpy()) / DAY) <= RECENT]
    meal_pop = np.asarray(K.sum(axis=0)).ravel()
    meal_pop_recent = np.bincount(k_recent["m"].to_numpy(), minlength=nm)

    user_all = np.bincount(uu, minlength=nu)
    user_recent = np.bincount(uu[recent], minlength=nu)
    last = pd.Series(age).groupby(uu).min().reindex(range(nu)).to_numpy(np.float32)
    return State(
        reviewed=R.toarray() > 0,
        log_days_since=np.log1p(days),
        cf=cf,
        in_known=((K @ cat.meal_course) > 0).toarray(),
        log_pop=np.log1p(pop),
        log_pop_recent=np.log1p(pop_recent),
        meal_log_pop=np.log1p(meal_pop),
        meal_log_pop_recent=np.log1p(meal_pop_recent),
        user={"log_reviews": np.log1p(user_all), "log_reviews_recent": np.log1p(user_recent),
              "log_days_since_last": np.log1p(last),
              "log_known_meals": np.log1p(np.asarray(K.sum(axis=1)).ravel())},
        known=K,
    )


def meal_features(st: State, cat: Catalog) -> dict[str, np.ndarray]:
    """Feature arrays (each U x M, float32) for the catalog meals."""
    nu, nm = st.reviewed.shape[0], len(cat.triples)
    f: dict[str, np.ndarray] = {}
    for j, name in enumerate(("app", "main", "dessert")):
        col = cat.triples[:, j]
        f[f"{name}/reviewed"] = st.reviewed[:, col]
        f[f"{name}/log_days_since"] = st.log_days_since[:, col]
        f[f"{name}/cf"] = st.cf[:, col]
        f[f"{name}/in_known_meals"] = st.in_known[:, col]
        f[f"{name}/log_pop"] = np.broadcast_to(st.log_pop[col], (nu, nm))
        f[f"{name}/log_pop_recent"] = np.broadcast_to(st.log_pop_recent[col], (nu, nm))
    f["meal/n_reviewed"] = f["app/reviewed"].astype(np.float32) + f["main/reviewed"] + f["dessert/reviewed"]
    f["meal/n_in_known_meals"] = (f["app/in_known_meals"].astype(np.float32) + f["main/in_known_meals"]
                                  + f["dessert/in_known_meals"])
    f["meal/log_pop"] = np.broadcast_to(st.meal_log_pop, (nu, nm))
    f["meal/log_pop_recent"] = np.broadcast_to(st.meal_log_pop_recent, (nu, nm))
    for j, pair in enumerate(("app_main", "app_dessert", "main_dessert")):
        f[f"pair/{pair}_log_shared"] = np.broadcast_to(np.log1p(cat.pair_counts[:, j]), (nu, nm))
    for name, v in st.user.items():
        f[f"user/{name}"] = np.broadcast_to(v[:, None], (nu, nm))
    return {name: np.asarray(v, dtype=np.float32) for name, v in f.items()}


def features_at(t: np.datetime64, data: MealRecPlus, cat: Catalog, reviews: pd.DataFrame,
                completions: pd.DataFrame) -> tuple[dict[str, np.ndarray], sp.csr_matrix]:
    """Feature arrays (each U x M, float32) as of time t, and the U x M matrix of meals known by t."""
    st = state_at(t, data, cat, reviews, completions)
    return meal_features(st, cat), st.known
