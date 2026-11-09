"""Load MealRec+ (H or L) into sparse matrices.

Mirrors the upstream loader (WUT-IDEA/MealRecPlus, MealRec+/data_load.py) but
returns plain numpy/scipy objects. Run scripts/download_mealrecplus.sh first.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASETS = ("MealRec+H", "MealRec+L")
SPLITS = ("train", "tune", "test")

# (users, meals, courses), as hard-coded in the upstream loader.
SIZES = {
    "MealRec+H": (1575, 3817, 7280),
    "MealRec+L": (1928, 3578, 10589),
}


@dataclass
class MealRecPlus:
    name: str
    n_users: int
    n_meals: int
    n_courses: int
    user_course: sp.csr_matrix           # U x C, binary; ALL interactions (no split upstream)
    meal_course: sp.csr_matrix           # M x C, binary; every meal has 3 courses
    course_category: np.ndarray          # C, 0 appetizer / 1 main / 2 dessert; -1 if in no meal
    user_meal: dict[str, sp.csr_matrix]  # split -> U x M, binary
    course_fsa: np.ndarray
    course_who: np.ndarray
    meal_fsa: np.ndarray                 # 4 (healthy) .. 12 (unhealthy)
    meal_who: np.ndarray                 # 0 (unhealthy) .. 7 (healthy)
    user_fsa: np.ndarray                 # mean meal_fsa over the user's TRAIN meals (orig split)
    user_who: np.ndarray
    cutoffs: dict | None = None          # set by temporal protocols (mealrec.protocols)

    @property
    def root(self) -> Path:
        return DATA_ROOT / self.name


def read_pairs(path: Path) -> np.ndarray:
    pairs = np.loadtxt(path, dtype=np.int64, delimiter="\t", ndmin=2)
    return pairs.reshape(-1, 2)


def read_scores(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64, ndmin=1)


def to_matrix(pairs: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    m = sp.csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (pairs[:, 0], pairs[:, 1])), shape=shape
    )
    m.data[:] = 1.0  # collapse duplicate pairs
    return m


BUNDLE_ROOT = DATA_ROOT / "bundle_crosscbr" / "datasets"
BUNDLE_DATASETS = ("Youshu", "NetEase", "iFashion")


@dataclass
class BundleData:
    """A generic bundle dataset in the CrossCBR layout, with MealRecPlus field names.

    meal = bundle, course = item, so mealrec.baselines and mealrec.protocols.mask_heldout
    apply unchanged. user_item.txt is unsplit upstream, as user_course.txt is in MealRec+.
    """
    name: str
    n_users: int
    n_meals: int
    n_courses: int
    user_course: sp.csr_matrix
    meal_course: sp.csr_matrix
    user_meal: dict[str, sp.csr_matrix]


def load_bundle(name: str, root: Path = BUNDLE_ROOT) -> BundleData:
    """Youshu, NetEase or iFashion from mysbupt/CrossCBR dataset.tgz (see data/README.md)."""
    d = root / name
    if not d.exists():
        raise FileNotFoundError(f"{d} not found; extract CrossCBR's dataset.tgz into {root.parent}")
    nu, nb, ni = (int(x) for x in (d / f"{name}_data_size.txt").read_text().split()[:3])
    return BundleData(
        name=name,
        n_users=nu,
        n_meals=nb,
        n_courses=ni,
        user_course=to_matrix(read_pairs(d / "user_item.txt"), (nu, ni)),
        meal_course=to_matrix(read_pairs(d / "bundle_item.txt"), (nb, ni)),
        user_meal={s: to_matrix(read_pairs(d / f"user_bundle_{s}.txt"), (nu, nb)) for s in SPLITS},
    )


def load(name: str, root: Path = DATA_ROOT) -> MealRecPlus:
    if name not in SIZES:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    d = root / name
    if not d.exists():
        raise FileNotFoundError(f"{d} not found; run scripts/download_mealrecplus.sh")
    nu, nm, nc = SIZES[name]

    mc_pairs = read_pairs(d / "meal_course.txt")
    # course_category.txt is one line per meal-course affiliation, not per course.
    category = np.full(nc, -1, dtype=np.int64)
    cat_pairs = read_pairs(d / "course_category.txt")
    category[cat_pairs[:, 0]] = cat_pairs[:, 1]

    h = d / "healthiness"
    data = MealRecPlus(
        name=name,
        n_users=nu,
        n_meals=nm,
        n_courses=nc,
        user_course=to_matrix(read_pairs(d / "user_course.txt"), (nu, nc)),
        meal_course=to_matrix(mc_pairs, (nm, nc)),
        course_category=category,
        user_meal={s: to_matrix(read_pairs(d / f"user_meal_{s}.txt"), (nu, nm)) for s in SPLITS},
        course_fsa=read_scores(h / "course_fsa.txt"),
        course_who=read_scores(h / "course_who.txt"),
        meal_fsa=read_scores(h / "meal_fsa.txt"),
        meal_who=read_scores(h / "meal_who.txt"),
        user_fsa=read_scores(h / "user_fsa.txt"),
        user_who=read_scores(h / "user_who.txt"),
    )
    for field, n in [("course_fsa", nc), ("meal_fsa", nm), ("user_fsa", nu)]:
        if len(getattr(data, field)) != n:
            raise ValueError(f"{name}: {field} has {len(getattr(data, field))} rows, expected {n}")
    return data
