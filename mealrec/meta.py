"""Read the Git LFS metadata (course.csv, user_course.csv) and align it to indices.

Needs `bash scripts/download_mealrecplus.sh --meta`. course.csv is ~700 MB
uncompressed (it carries reviews), so load only the columns you need.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from mealrec.data import DATA_ROOT


def read_index(path: Path) -> dict[int, int]:
    pairs = np.loadtxt(path, dtype=np.int64, delimiter="\t", ndmin=2)
    return dict(zip(pairs[:, 0].tolist(), pairs[:, 1].tolist()))


def user_course_meta(name: str, root: Path = DATA_ROOT) -> pd.DataFrame:
    """Columns: u, c (indices), rating, t (timestamp of the review's last edit)."""
    d = root / name / "meta_data"
    df = pd.read_csv(d / "user_course.zip")
    df["u"] = df["user_id"].map(read_index(d / "user2index.txt"))
    df["c"] = df["course_id"].map(read_index(d / "course2index.txt"))
    df["t"] = pd.to_datetime(df["dateLastModified"].str.strip(), format="mixed")
    return df[["u", "c", "rating", "t"]]


def course_meta(name: str, columns=("course_name", "category"), root: Path = DATA_ROOT) -> pd.DataFrame:
    """course.csv indexed by course index; `nutritions` is parsed to a dict if requested."""
    d = root / name / "meta_data"
    df = pd.read_csv(d / "course.zip", usecols=["course_id", *columns])
    df.index = df.pop("course_id").map(read_index(d / "course2index.txt"))
    if "nutritions" in df:
        df["nutritions"] = df["nutritions"].map(ast.literal_eval)
    return df.sort_index()


def nutrient(nutritions: pd.Series, key: str) -> np.ndarray:
    """Per-serving amount of one nutrient (NaN if missing)."""
    def get(n):
        try:
            return float(n[key]["amount"])
        except (KeyError, TypeError, ValueError):
            return np.nan
    return nutritions.map(get).to_numpy()
