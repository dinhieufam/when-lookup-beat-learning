"""G4a: open composition. Can a model propose the held-out meals without being shown them?

Setting: the cold_p1 split (meals split 7:1:2, P1 mask on user_course). The model knows the
training meals only; it is never given the tune/test meals as candidates. It generates
(appetizer, main, dessert) triples over every course of each category in course.csv (the
recipe metadata, independent of the split; it agrees 100% with the meal categories), e.g.
2,737 x 2,552 x 1,991 on MealRec+H. A generated triple is a hit if it is one of the user's
held-out test meals.

- CompMeal-N (mealrec.compmeal_nn) in that course space, trained by the cold_p1 fold
  simulation; course-pair counts only from meals known in each state (training meals at
  test time). Top-k by beam search over slots (open_topk). Two trainings:
  catalog-trained: catalog negatives only, epoch by catalog tune R@20 (as in G3);
  open-trained (train_open): also random and recombined-course triples as negatives, epoch
  by open tune R@20 (tune meals are cold too, and never shown to the model).
- SlotPop: the most popular courses in training meals, same triples for every user.
- TrainCover-open: courses in the user's training meals first (by popularity), no pairs.
- Reference: the same CompMeal-N ranking the true catalog (cold meals given), i.e. cold_p1.

    python -m experiments.g4_open_composition [--seeds 0 1 2] [--beam 40]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from experiments.b1_compmeal_static import N_FOLDS, held_out_fold
from experiments.b1n_compmeal_nn import LR, WD, state_for
from mealrec.compmeal import PAIRS, catalog, pair_count_matrix
from mealrec.compmeal_nn import content_embedding, train_compmeal_nn, train_open
from mealrec.data import DATASETS, load
from mealrec.meta import course_meta, user_course_meta
from mealrec.protocols import completion_times, get, mask_heldout

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "g4_open_composition.json"
CATEGORY = {"appetizer": 0, "appetizers": 0, "main-dish": 1, "dessert": 2, "desserts": 2}
KS = (20, 80)


def open_metrics(tops: dict[int, list[tuple]], test: sp.csr_matrix, triple_of: np.ndarray) -> dict:
    """Exact-meal recall@k and per-slot course recall@k of generated triples (used-space ids)."""
    out = {f"recall@{k}": [] for k in KS} | {f"slot_recall@{k}": [] for k in KS}
    for u, top in tops.items():
        pos = {tuple(t) for t in triple_of[test[u].indices]}
        for k in KS:
            gen = top[:k]
            out[f"recall@{k}"].append(len(pos & set(gen)) / len(pos))
            hit = [np.mean([p[j] in {g[j] for g in gen} for p in pos]) for j in range(3)]
            out[f"slot_recall@{k}"].append(float(np.mean(hit)))
    return {m: float(np.mean(v)) for m, v in out.items()}


def filter_known(cands: np.ndarray, known: set, k: int) -> list[tuple]:
    out = [tuple(int(x) for x in c) for c in cands if tuple(int(x) for x in c) not in known]
    return out[:k]


def slot_beam(scores: list[np.ndarray], k: int, beam: int) -> np.ndarray:
    """Top-k triples of an additive score without pair terms; scores[j]: positions in slot j."""
    idx = [np.argsort(-s, kind="stable")[:beam] for s in scores]
    total = scores[0][idx[0]][:, None, None] + scores[1][idx[1]][None, :, None] + scores[2][idx[2]][None, None, :]
    flat = np.argsort(-total.ravel(), kind="stable")[:k]
    a, b, c = np.unravel_index(flat, total.shape)
    return np.stack([idx[0][a], idx[1][b], idx[2][c]], axis=1)


def dense_pairs(cat, meals: np.ndarray, slot_courses, device) -> list[torch.Tensor]:
    out = []
    for j, (sa, sb) in enumerate(PAIRS):
        m = pair_count_matrix(cat, j, meals)[slot_courses[sa]][:, slot_courses[sb]]
        out.append(torch.from_numpy(np.log1p(m.toarray()).astype(np.float32)).to(device))
    return out


def generate(model, ev, users, dense, slot_t, k_gen, beam, known_triples, device, k_keep) -> dict:
    tops = {}
    for i in range(0, len(users), 32):
        ub = users[i:i + 32]
        gen = model.open_topk(ev, torch.from_numpy(ub).to(device), dense, slot_t, k_gen, beam).cpu().numpy()
        for u, g in zip(ub, gen):
            tops[u] = filter_known(g, known_triples[u], k_keep)
    return tops


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--beam", type=int, default=40)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in a.datasets:
        t_start = time.time()
        base = load(name)
        uc = user_course_meta(name)
        meta = course_meta(name, columns=("course_name", "category", "ingredients", "tags"))
        category = meta["category"].map(CATEGORY).reindex(range(base.n_courses))
        used = np.flatnonzero(category.notna().to_numpy())
        slot_of = category.to_numpy()[used].astype(np.int64)
        comps = completion_times(base, uc).drop_duplicates(["u", "m"])
        content = content_embedding(meta, used)
        data = get("cold_p1", name)
        train, tune, test = (data.user_meal[s] for s in ("train", "tune", "test"))
        train_meals = train.getnnz(axis=0) > 0
        cat = catalog(base, used=used, pair_meals=train_meals)
        t_end = np.datetime64(uc["t"].max())

        rng = np.random.default_rng(0)
        sets = []
        for _ in range(N_FOLDS):
            held = held_out_fold("cold_p1", train, rng)
            known = sp.csr_matrix(train - held)
            known.eliminate_zeros()
            pseudo = mask_heldout(replace(data, user_meal={"train": known, "tune": held, "test": held * 0}))
            fold_meals = known.getnnz(axis=0) > 0
            fold_cat = catalog(base, used=used, pair_meals=fold_meals)
            sets.append((state_for(pseudo, known, uc, cat, comps, t_end), held, fold_cat.pair_counts, fold_meals))
        ev_state = state_for(data, train, uc, cat, comps, t_end)
        print(f"{name}: course space {len(used)} ({np.bincount(slot_of).tolist()}), "
              f"{int(train_meals.sum())} training meals", flush=True)

        # Slot positions and pair counts over training meals, for open generation.
        slot_courses = [np.flatnonzero(slot_of == j) for j in range(3)]
        dense = dense_pairs(cat, train_meals, slot_courses, device)
        slot_t = [torch.from_numpy(s).to(device) for s in slot_courses]

        users = np.flatnonzero(test.getnnz(axis=1))
        tune_users = np.flatnonzero(tune.getnnz(axis=1))
        known_triples = {u: {tuple(t) for t in cat.triples[train[u].indices]} for u in np.union1d(users, tune_users)}
        max_known = max(len(v) for v in known_triples.values())
        open_sets = [(st, held, dense_pairs(cat, fm, slot_courses, device)) for st, held, _, fm in sets]
        cat_sets = [(st, held, pc) for st, held, pc, _ in sets]
        k_gen = max(KS) + max_known

        # Baselines in the open space.
        course_pop = np.asarray((train @ cat.meal_course).sum(axis=0)).ravel()
        tie = np.log1p(course_pop) / np.log1p(course_pop).max() * 1e-3
        mine = ((train @ cat.meal_course) > 0).toarray()
        pop_top = slot_beam([tie[s] for s in slot_courses], k_gen, a.beam)
        to_used = lambda tri: np.stack([slot_courses[j][tri[:, j]] for j in range(3)], axis=1)
        baselines = {
            "SlotPop": {u: filter_known(to_used(pop_top), known_triples[u], max(KS)) for u in users},
            "TrainCover-open": {u: filter_known(to_used(slot_beam([mine[u, s] + tie[s] for s in slot_courses],
                                                                   k_gen, a.beam)), known_triples[u], max(KS))
                                for u in users},
        }
        block = {"course_space": [int(len(s)) for s in slot_courses], "beam": a.beam,
                 "baselines": {b: open_metrics(t, test, cat.triples) for b, t in baselines.items()}, "seeds": []}
        for b, m in block["baselines"].items():
            print(f"{name} {b:<16} R@20 {m['recall@20']:.4f}  R@80 {m['recall@80']:.4f}  slot R@20 {m['slot_recall@20']:.4f}",
                  flush=True)

        def tune_open(model, ev):
            tops = generate(model, ev, tune_users, dense, slot_t, k_gen, a.beam, known_triples, device, 20)
            return open_metrics(tops, tune, cat.triples)["recall@20"]

        for seed in a.seeds:
            r = train_compmeal_nn(cat_sets, ev_state, tune, test, train, cat, slot_of, content, seed, device,
                                  use_prior=False, lr=LR, weight_decay=WD, log=lambda *_: None)
            row = {"seed": seed, "catalog_trained": {
                "epoch": r["epoch"], "catalog_test": r["test"],
                "open": open_metrics(generate(r["model"], r["eval_tensors"], users, dense, slot_t, k_gen, a.beam,
                                              known_triples, device, max(KS)), test, cat.triples)}}
            o = train_open(open_sets, ev_state, cat, slot_of, content, slot_courses, tune_open, seed, device,
                           lr=LR, weight_decay=WD, log=lambda *_: None)
            row["open_trained"] = {"epoch": o["epoch"], "tune_curve": o["tune_curve"],
                                   "open": open_metrics(generate(o["model"], o["eval_tensors"], users, dense, slot_t,
                                                                 k_gen, a.beam, known_triples, device, max(KS)),
                                                        test, cat.triples)}
            block["seeds"].append(row)
            for kind in ("catalog_trained", "open_trained"):
                m = row[kind]["open"]
                print(f"{name} CompMeal-N {kind:<15} seed {seed}: open R@20 {m['recall@20']:.4f}  R@80 {m['recall@80']:.4f}"
                      f"  slot R@20 {m['slot_recall@20']:.4f}", flush=True)
            print(f"{name} catalog (cold meals given) R@20 {r['test']['recall@20']:.4f}", flush=True)
        for kind in ("catalog_trained", "open_trained"):
            for key in ("recall@20", "recall@80", "slot_recall@20"):
                v = [s[kind]["open"][key] for s in block["seeds"]]
                block[f"{kind}/open_{key}_mean"] = float(np.mean(v))
                block[f"{kind}/open_{key}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        block["catalog_recall@20_mean"] = float(np.mean([s["catalog_trained"]["catalog_test"]["recall@20"]
                                                         for s in block["seeds"]]))
        block["seconds"] = time.time() - t_start
        result[name] = block
        OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
