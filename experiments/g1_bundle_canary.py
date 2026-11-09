"""G1: does the MealRec+ leak generalise to the standard bundle benchmarks?

Youshu, NetEase and iFashion (CrossCBR's release) ship an unsplit user_item.txt, as
MealRec+ ships an unsplit user_course.txt. For each dataset (MealRec+H/L included for
reference) we measure how much user_item reveals about held-out bundles and run the
training-free baselines with full ranking, scored in user chunks so iFashion fits in memory.

    python -m experiments.g1_bundle_canary
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from mealrec.data import BUNDLE_DATASETS, DATASETS, SPLITS, load, load_bundle
from mealrec.metrics import per_user_metrics, topk
from mealrec.protocols import mask_heldout

OUT = Path(__file__).resolve().parent.parent / "results" / "g1_bundle_canary.json"
CHUNK = 2048
KS = (20, 40, 80)
RANDOM_PAIRS = 200_000

# Published numbers on the same splits, Recall@20 / NDCG@20.
# Youshu/NetEase/iFashion: CrossCBR (Ma et al., KDD 2022, Table 2) and MultiCBR (Ma et al., TOIS 2024,
# arXiv 2311.16751, Table 2). MealRec+: CrossCBR in Li et al., SIGIR 2024, Table 6 (no MultiCBR row).
REPORTED = {
    "Youshu": {"CrossCBR": {"recall@20": 0.2813, "ndcg@20": 0.1668},
               "MultiCBR": {"recall@20": 0.2842, "ndcg@20": 0.1693}},
    "NetEase": {"CrossCBR": {"recall@20": 0.0842, "ndcg@20": 0.0457},
                "MultiCBR": {"recall@20": 0.0907, "ndcg@20": 0.0494}},
    "iFashion": {"CrossCBR": {"recall@20": 0.1173, "ndcg@20": 0.0895},
                 "MultiCBR": {"recall@20": 0.1500, "ndcg@20": 0.1203}},
    "MealRec+H": {"CrossCBR": {"recall@20": 0.3753, "ndcg@20": 0.2588}},
    "MealRec+L": {"CrossCBR": {"recall@20": 0.1920, "ndcg@20": 0.1018}},
}


def pair_coverage(user_items: sp.csr_matrix, bundle_items: sp.csr_matrix, users, bundles) -> np.ndarray:
    """Share of each bundle's items that the user has in user_items, for aligned (user, bundle) pairs."""
    hit = np.asarray(user_items[users].multiply(bundle_items[bundles]).sum(axis=1)).ravel()
    return hit / np.asarray(bundle_items[bundles].sum(axis=1)).ravel()


def leak_stats(data, rng: np.random.Generator) -> dict:
    ui, bi = data.user_course, data.meal_course
    sizes = np.asarray(bi.sum(axis=1)).ravel()
    out = {
        "users_bundles_items": [data.n_users, data.n_meals, data.n_courses],
        "pairs": {s: int(data.user_meal[s].nnz) for s in SPLITS},
        "bundle_size": {"mean": float(sizes.mean()), "median": float(np.median(sizes))},
        "coverage_given_positive": {},
    }
    for s in SPLITS:
        u, b = data.user_meal[s].nonzero()
        cov = pair_coverage(ui, bi, u, b)
        out["coverage_given_positive"][s] = {"mean": float(cov.mean()), "p_full": float((cov >= 1).mean()),
                                             "p_any": float((cov > 0).mean())}
    u = rng.integers(0, data.n_users, RANDOM_PAIRS)
    b = rng.integers(0, data.n_meals, RANDOM_PAIRS)
    cov = pair_coverage(ui, bi, u, b)
    out["coverage_random_pair"] = {"mean": float(cov.mean()), "p_full": float((cov >= 1).mean()),
                                   "p_any": float((cov > 0).mean())}

    # How much of user_item is explained by the user's own bundles, and by held-out bundles only.
    def reach(m):
        r = sp.csr_matrix(m @ bi, dtype=np.float32)
        r.data[:] = 1.0
        return r
    train_r = reach(data.user_meal["train"])
    held_r = reach(data.user_meal["tune"] + data.user_meal["test"])
    any_r = reach(data.user_meal["train"] + data.user_meal["tune"] + data.user_meal["test"])
    n = ui.nnz
    in_any = ui.multiply(any_r).nnz
    only_held = ui.multiply(held_r).nnz - ui.multiply(held_r).multiply(train_r).nnz
    out["user_item_edges"] = int(n)
    out["share_edges_in_own_bundles"] = in_any / n
    out["share_edges_only_in_heldout_bundles"] = only_held / n
    return out


def scorers(data) -> dict:
    """Each entry maps a user index array to a dense (len(users) x bundles) score block."""
    bi = data.meal_course
    sizes = np.asarray(bi.sum(axis=1)).ravel()
    pop = np.asarray(data.user_meal["train"].sum(axis=0)).ravel()
    tie = 1e-3 * pop / max(pop.max(), 1)
    train_items = sp.csr_matrix(data.user_meal["train"] @ bi, dtype=np.float32)
    train_items.data[:] = 1.0
    masked_ui = mask_heldout(data).user_course

    def cover(ui):
        return lambda users: (ui[users] @ bi.T).toarray() / sizes[None, :] + tie[None, :]

    return {
        "MostPop": lambda users: np.tile(pop.astype(np.float64), (len(users), 1)),
        "TrainCover": cover(train_items),
        "CoverHeuristic": cover(data.user_course),
        "CoverHeuristic_p1": cover(masked_ui),
    }


def evaluate(data, score_fn) -> dict:
    test, train = data.user_meal["test"], data.user_meal["train"]
    users = np.flatnonzero(test.getnnz(axis=1))
    parts = {f"{m}@{k}": [] for k in KS for m in ("recall", "ndcg")}
    for i in range(0, len(users), CHUNK):
        us = users[i:i + CHUNK]
        s = np.asarray(score_fn(us), dtype=np.float64)
        r, c = train[us].nonzero()
        s[r, c] = -np.inf
        for key, v in per_user_metrics(topk(s, max(KS)), test[us], KS).items():
            parts[key].append(v)
    return {key: float(np.concatenate(v).mean()) for key, v in parts.items()}


def main() -> None:
    rng = np.random.default_rng(0)
    results = {}
    for name in BUNDLE_DATASETS + DATASETS:
        data = load_bundle(name) if name in BUNDLE_DATASETS else load(name)
        row = {"leakage": leak_stats(data, rng), "reported": REPORTED[name], "baselines": {}}
        lk = row["leakage"]
        print(f"\n== {name} ==  bundle size mean {lk['bundle_size']['mean']:.1f}")
        print(f"coverage | test positive: mean {lk['coverage_given_positive']['test']['mean']:.3f}, "
              f"full {lk['coverage_given_positive']['test']['p_full']:.3f}; "
              f"random pair: mean {lk['coverage_random_pair']['mean']:.4f}")
        print(f"user_item edges in own bundles {lk['share_edges_in_own_bundles']:.3f}, "
              f"only in held-out bundles {lk['share_edges_only_in_heldout_bundles']:.3f}")
        for model, fn in scorers(data).items():
            acc = evaluate(data, fn)
            row["baselines"][model] = acc
            print(f"{model:<18} R@20 {acc['recall@20']:.4f}  N@20 {acc['ndcg@20']:.4f}  R@80 {acc['recall@80']:.4f}")
        for model, rep in REPORTED[name].items():
            print(f"{model + ' (paper)':<18} R@20 {rep['recall@20']:.4f}  N@20 {rep['ndcg@20']:.4f}")
        results[name] = row
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
