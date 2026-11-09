"""Read the outputs of experiments/train_bundle.py (results/runs/<protocol>/<dataset>/<model>/seed<k>/)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
RUNS = RESULTS / "runs"
MODELS = ("LightGCN", "BGCN", "CrossCBR", "MultiCBR", "HyperMBR")
COURSE_AWARE = ("BGCN", "CrossCBR", "MultiCBR", "HyperMBR")


def run_dir(protocol: str, dataset: str, model: str, seed: int, root: str = "runs") -> Path:
    """`model` may carry a setting tag, as in results/runs_tuned/: "CrossCBR__lr0.001_l21e-06_L1"."""
    return RESULTS / root / protocol / dataset / model / f"seed{seed}"


def summaries(protocol: str | None = None, root: str = "runs") -> pd.DataFrame:
    """One row per finished run under results/<root>/ with test metrics under both epoch-selection rules."""
    rows = []
    for path in sorted((RESULTS / root).glob(f"{protocol or '*'}/*/*/seed*/summary.json")):
        s = json.loads(path.read_text())
        row = {k: s[k] for k in ("protocol", "dataset", "model", "seed", "epochs_run",
                                 "tune_best_epoch", "test_best_epoch", "seconds", "sec_per_epoch")}
        row["tag"] = s.get("tag", "")
        row["lr"], row["l2"], row["layers"] = s["args"].get("lr"), s["args"].get("l2"), s["args"]["layers"]
        for rule in ("tune", "test"):
            h = s[f"select_by_{rule}"]
            for m, v in h["test"].items():
                row[f"{rule}_sel/{m}"] = v
            row[f"{rule}_sel/tune_recall@20"] = h["tune"]["recall@20"]
        rows.append(row)
    return pd.DataFrame(rows)


def per_user(protocol: str, dataset: str, model: str, seed: int, selection: str = "tune_best",
             split: str = "test", root: str = "runs") -> dict[str, np.ndarray]:
    """Per-user metrics (and the `users` index) at the epoch chosen by `selection`."""
    z = np.load(run_dir(protocol, dataset, model, seed, root) / "per_user.npz")
    prefix = f"{selection}/{split}/"
    out = {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}
    out["users"] = out["users"].astype(np.int64)  # early runs stored the index as float32 (exact)
    return out


def scores(protocol: str, dataset: str, model: str, seed: int) -> np.ndarray:
    return np.load(run_dir(protocol, dataset, model, seed) / "scores_tune_best.npy").astype(np.float32)


def mean_sd(df: pd.DataFrame, cols: list[str], by=("dataset", "model")) -> pd.DataFrame:
    g = df.groupby(list(by))[cols]
    out = g.mean().add_suffix("/mean").join(g.std(ddof=1).add_suffix("/sd")).join(g.size().rename("n_seeds"))
    return out.reset_index()
