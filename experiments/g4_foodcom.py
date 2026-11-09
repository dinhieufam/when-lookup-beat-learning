"""G4a on Food.com: open composition with real sessions (mealrec.foodcom).

For each tune/test session, rank every recipe of each slot that was submitted before the
session start and that the user has not reviewed before, from what is known at the session
start: the user's own reviews before the start, and global statistics (popularity,
co-review similarity) from reviews before the split boundary (train_until for tune
sessions, tune_until for test sessions). Then compose triples by beam search over slots.

Metrics per session, averaged over sessions:
- slot_recall@k: share of the session's recipes in a slot found in that slot's top-k,
  averaged over the three slots;
- meal_hit@k: 1 if one of the top-k triples has all three recipes in the session.

    python -m experiments.g4_foodcom [--window 1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mealrec import foodcom

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "g4_foodcom.json"
KS = (20, 100)
BEAM = 30
RECENT = 365


class Context:
    """Global statistics as of a boundary date, over all recipes."""

    def __init__(self, fc: foodcom.FoodCom, boundary: np.datetime64, rid: pd.Index, uid: pd.Index):
        r = fc.reviews[fc.reviews["date"] < boundary]
        rows, cols = uid.get_indexer(r["user_id"]), rid.get_indexer(r["recipe_id"])
        x = sp.csr_matrix((np.ones(len(r), np.float32), (rows, cols)), shape=(len(uid), len(rid)))
        x.data[:] = 1
        deg = np.asarray(x.sum(axis=0)).ravel()
        self.xn = (x @ sp.diags(1 / np.sqrt(np.maximum(deg, 1)))).tocsr()   # users x recipes, column-normalised
        self.xnt = self.xn.T.tocsr()
        self.pop = deg
        recent = r[r["date"] >= boundary - RECENT * foodcom.DAY]
        self.pop_recent = np.bincount(rid.get_indexer(recent["recipe_id"]), minlength=len(rid)).astype(float)

    def knn(self, hist: np.ndarray) -> np.ndarray:
        """sum over history recipes h of cos(h, r), for every recipe r."""
        h = sp.csr_matrix((np.ones(len(hist), np.float32), (np.zeros(len(hist), int), hist)), shape=(1, self.xn.shape[1]))
        return np.asarray(((h @ self.xnt) @ self.xn).todense()).ravel()


def beam_triples(scores: list[np.ndarray], cands: list[np.ndarray], k: int) -> list[tuple]:
    """Top-k triples of sum of per-slot scores (each standardised over its candidates)."""
    top = []
    for s, c in zip(scores, cands):
        z = (s - s.mean()) / (s.std() + 1e-9)
        o = np.argsort(-z, kind="stable")[:BEAM]
        top.append((c[o], z[o]))
    total = top[0][1][:, None, None] + top[1][1][None, :, None] + top[2][1][None, None, :]
    flat = np.argsort(-total.ravel(), kind="stable")[:k]
    a, b, d = np.unravel_index(flat, total.shape)
    return list(zip(top[0][0][a], top[1][0][b], top[2][0][d]))


def evaluate(fc, split: str, scorer, rid, slot_arr, submitted) -> dict:
    s = fc.sessions[fc.sessions["split"] == split]
    out = {f"slot_recall@{k}": [] for k in KS} | {f"meal_hit@{k}": [] for k in KS}
    rev = fc.reviews.sort_values("date")
    by_user = {u: g for u, g in rev.groupby("user_id")}
    for sess, g in s.groupby("session"):
        user, start = g["user_id"].iloc[0], g["start"].iloc[0]
        h = by_user.get(user)
        hist = rid.get_indexer(h.loc[h["date"] < start, "recipe_id"]) if h is not None else np.array([], int)
        hist = hist[hist >= 0]
        allowed = submitted < start
        allowed[hist] = False
        full = scorer(user, start, hist)
        truth = [set(rid.get_indexer(g.loc[g["slot"] == j, "recipe_id"])) for j in range(3)]
        scores, cands = [], []
        for j in range(3):
            c = np.flatnonzero(allowed & (slot_arr == j))
            scores.append(full[c])
            cands.append(c)
            for k in KS:
                if j == 0:
                    out[f"slot_recall@{k}"].append([])
                top = set(c[np.argsort(-full[c], kind="stable")[:k]])
                out[f"slot_recall@{k}"][-1].append(len(top & truth[j]) / len(truth[j]))
        tri = beam_triples(scores, cands, max(KS))
        for k in KS:
            out[f"meal_hit@{k}"].append(float(any(a in truth[0] and m in truth[1] and d in truth[2] for a, m, d in tri[:k])))
    res = {m: float(np.mean([np.mean(v) if isinstance(v, list) else v for v in vals])) for m, vals in out.items()}
    res["sessions"] = int(s["session"].nunique())
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=1)
    a = p.parse_args()
    fc = foodcom.load(window=a.window)
    rid = pd.Index(fc.recipes.index)
    uid = pd.Index(fc.reviews["user_id"].unique())
    slot_arr = np.full(len(rid), -1)
    slot_arr[rid.get_indexer(fc.slot.index)] = fc.slot.to_numpy()
    submitted = pd.to_datetime(fc.recipes["submitted"]).to_numpy().astype("datetime64[D]")
    result = {"window": a.window, "cutoffs": fc.cutoffs, "baselines": {}}
    for split, boundary in (("tune", fc.cutoffs["train_until"]), ("test", fc.cutoffs["tune_until"])):
        ctx = Context(fc, np.datetime64(pd.Timestamp(boundary).date()), rid, uid)
        scorers = {
            "SlotPop": lambda u, t, h: ctx.pop_recent + 1e-3 * ctx.pop,
            "ItemKNN": lambda u, t, h: ctx.knn(h) + 1e-6 * ctx.pop_recent,
        }
        for name, fn in scorers.items():
            r = evaluate(fc, split, fn, rid, slot_arr, submitted)
            result["baselines"].setdefault(name, {})[split] = r
            print(f"w={a.window} {split} {name:<8} " + "  ".join(f"{k} {v:.4f}" for k, v in r.items() if k != "sessions")
                  + f"  ({r['sessions']} sessions)", flush=True)
    out = OUT.with_name(f"g4_foodcom_w{a.window}.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
