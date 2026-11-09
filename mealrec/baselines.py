"""Training-free baselines. Each returns a dense U x M score matrix."""
from __future__ import annotations

import numpy as np

from mealrec.data import MealRecPlus


def most_pop(data: MealRecPlus) -> np.ndarray:
    """Meal popularity in the training split, identical for every user."""
    pop = np.asarray(data.user_meal["train"].sum(axis=0)).ravel()
    return np.tile(pop, (data.n_users, 1)).astype(np.float64)


def meal_item_knn(data: MealRecPlus, neighbours: int = 100) -> np.ndarray:
    """Item-based CF on training user-meal interactions only (cosine, top-n neighbours)."""
    um = data.user_meal["train"].astype(np.float64)
    norms = np.sqrt(np.asarray(um.multiply(um).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    sim = (um.T @ um).toarray() / np.outer(norms, norms)
    np.fill_diagonal(sim, 0.0)
    if neighbours < sim.shape[0]:
        cut = np.partition(sim, -neighbours, axis=1)[:, -neighbours][:, None]
        sim[sim < cut] = 0.0
    return np.asarray(um @ sim)


def course_coverage(data: MealRecPlus) -> np.ndarray:
    """Number of a meal's courses the user has interacted with (U x M)."""
    return np.asarray((data.user_course @ data.meal_course.T).todense())


def cover_heuristic(data: MealRecPlus) -> np.ndarray:
    """Leak oracle: fraction of the meal's courses in user_course.txt, ties by popularity.

    user_course.txt is not split, so it contains the courses of test meals.
    """
    sizes = np.asarray(data.meal_course.sum(axis=1)).ravel()
    pop = np.asarray(data.user_meal["train"].sum(axis=0)).ravel()
    return course_coverage(data) / sizes[None, :] + 1e-3 * pop[None, :] / max(pop.max(), 1)


def train_cover(data: MealRecPlus) -> np.ndarray:
    """Legitimate counterpart of CoverHeuristic: coverage by courses of the user's TRAIN meals only.

    Uses nothing beyond what every model is given (train user-meal + meal-course), so it is
    clean under any protocol. MealRec+ labels a meal positive when all its courses were
    touched, so this alone recovers many held-out positives.
    """
    train_courses = ((data.user_meal["train"] @ data.meal_course) > 0).astype(np.float32)
    coverage = np.asarray((train_courses @ data.meal_course.T).todense())
    sizes = np.asarray(data.meal_course.sum(axis=1)).ravel()
    pop = np.asarray(data.user_meal["train"].sum(axis=0)).ravel()
    return coverage / sizes[None, :] + 1e-3 * pop[None, :] / max(pop.max(), 1)


BASELINES = {
    "MostPop": most_pop,
    "MealItemKNN": meal_item_knn,
    "TrainCover": train_cover,
    "CoverHeuristic": cover_heuristic,
}

