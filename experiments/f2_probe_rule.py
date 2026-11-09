"""F2: the lookup-probe rule with a test, and a check for a reverse footprint of the P1 mask.

For each dataset (MealRec+H/L, Youshu, NetEase, iFashion) and protocol (released, P1), per user:
- MostPop and TrainCover: the legitimate lookups (training data only);
- Probe: the lookup probe (CoverHeuristic) over the protocol's user-item graph.
The rule: a protocol passes when the probe does not significantly exceed the best legitimate
lookup, max(MostPop, TrainCover) by mean test Recall@20 (one-sided paired Wilcoxon over users,
alpha = 0.01). A pass is necessary, not sufficient, for a clean protocol.

Reverse footprint (P1 only). P1 removes edge (u, i) for items of u's held-out bundles unless
one of u's training bundles contains i. So for a held-out positive b, every visible item of b is
also a training-bundle item of u: its "extra" coverage (visible items outside u's training
bundles) is exactly 0, while a negative may have extra > 0. Footprint ranks by TrainCover and
demotes bundles with extra > 0. If it beats the best legitimate lookup, the mask itself leaks labels.

Per-user Recall@20 arrays go to results/runs_f2/ for the bootstrap in f3. CPU, ~15 minutes.

    python -m experiments.f2_probe_rule
"""
from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp
from scipy.stats import wilcoxon

from mealrec.data import BUNDLE_DATASETS, DATASETS, load, load_bundle
from mealrec.metrics import per_user_metrics, topk
from mealrec.protocols import mask_heldout
from mealrec.runs import RESULTS

OUT = RESULTS / "f2_probe_rule.json"
PER_USER = RESULTS / "runs_f2"
CHUNK = 2048
ALPHA = 0.01


def scorers(data, protocol: str) -> dict:
    bi = data.meal_course.tocsr()
    sizes = np.asarray(bi.sum(axis=1)).ravel()
    pop = np.asarray(data.user_meal["train"].sum(axis=0)).ravel().astype(np.float64)
    tie = 1e-3 * pop / max(pop.max(), 1)
    train_items = sp.csr_matrix(data.user_meal["train"] @ bi, dtype=np.float32)
    train_items.data[:] = 1.0
    ui = (mask_heldout(data).user_course if protocol == "p1" else data.user_course).tocsr().astype(np.float32)
    ui.data[:] = 1.0
    extra_items = (ui - ui.multiply(train_items)).tocsr()  # visible items outside u's training bundles
    extra_items.eliminate_zeros()

    def frac(m):
        return lambda users: (m[users] @ bi.T).toarray() / sizes[None, :]

    tc, probe, extra = frac(train_items), frac(ui), frac(extra_items)
    out = {
        "MostPop": lambda users: np.tile(pop, (len(users), 1)),
        "TrainCover": lambda users: tc(users) + tie[None, :],
        "Probe": lambda users: probe(users) + tie[None, :],
    }
    if protocol == "p1":
        # Demote any bundle with extra visible coverage below every bundle without it, keep TrainCover order.
        out["Footprint"] = lambda users: tc(users) + tie[None, :] - 10.0 * (extra(users) > 0)
    return out


def per_user_recall(data, score_fn) -> tuple[np.ndarray, np.ndarray]:
    test, train = data.user_meal["test"], data.user_meal["train"]
    users = np.flatnonzero(test.getnnz(axis=1))
    rec = []
    for i in range(0, len(users), CHUNK):
        us = users[i:i + CHUNK]
        s = np.asarray(score_fn(us), dtype=np.float64)
        r, c = train[us].nonzero()
        s[r, c] = -np.inf
        rec.append(per_user_metrics(topk(s, 20), test[us], (20,))["recall@20"])
    return users, np.concatenate(rec)


def one_sided(a: np.ndarray, b: np.ndarray) -> float:
    """p-value for mean(a) > mean(b), paired over users (zero differences dropped)."""
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b, alternative="greater").pvalue)


def main() -> None:
    PER_USER.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in DATASETS + BUNDLE_DATASETS:
        data = load_bundle(name) if name in BUNDLE_DATASETS else load(name)
        for protocol in ("orig", "p1"):
            recalls = {}
            for model, fn in scorers(data, protocol).items():
                users, recalls[model] = per_user_recall(data, fn)
                np.savez_compressed(PER_USER / f"{name}__{protocol}__{model}.npz", users=users, recall=recalls[model])
            legit = max(("MostPop", "TrainCover"), key=lambda m: recalls[m].mean())
            p = one_sided(recalls["Probe"], recalls[legit])
            row = {m: float(r.mean()) for m, r in recalls.items()}
            row.update({"best_legit": legit, "probe_minus_legit": row["Probe"] - row[legit],
                        "p_probe_gt_legit": p, "passes": bool(p >= ALPHA or row["Probe"] <= row[legit]),
                        "n_users": int(len(users))})
            if "Footprint" in recalls:
                # Compare with the best legitimate lookup: where TrainCover is mostly zero, Footprint
                # falls back to its popularity tie-break and equals MostPop.
                row["footprint_minus_legit"] = row["Footprint"] - row[legit]
                row["p_footprint_gt_legit"] = one_sided(recalls["Footprint"], recalls[legit])
            results[f"{name}/{protocol}"] = row
            print(f"{name:<10}{protocol:<5}" + "  ".join(f"{m} {v:.4f}" for m, v in row.items() if isinstance(v, float) and m in recalls)
                  + f"  | probe-{legit} {row['probe_minus_legit']:+.4f} p={p:.2g} {'PASS' if row['passes'] else 'FAIL'}"
                  + (f" | footprint-{legit} {row['footprint_minus_legit']:+.4f} p={row['p_footprint_gt_legit']:.2g}" if "Footprint" in recalls else ""), flush=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
