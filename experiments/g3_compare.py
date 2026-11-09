"""G3: CompMeal-N against the GBDT CompMeal and the best training-free lookup, per cell.

Per-user test R@20 is averaged over the 5 seeds of each model, then compared with a paired
Wilcoxon signed-rank test (Holm-corrected over the six cells, per comparison).
- CompMeal-N: results/runs_b1n/<dataset>/<protocol>/full/seed<k>.npz (b1n_compmeal_nn)
- GBDT: results/runs_b1/<dataset>/seed<k>.npz for P2 (b1_compmeal),
        results/runs_b1/<dataset>/<protocol>/seed<k>.npz for P1 / cold_p1 (b1_compmeal_static)
- Lookup: the better of TrainCover and CoverHeuristic on the protocol's test split.

    python -m experiments.g3_compare
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import wilcoxon

from experiments.b1n_compmeal_nn import CELLS
from experiments.r1_reviewer_checks import baseline_per_user
from mealrec.protocols import get, temporal
from mealrec.data import load
from mealrec.meta import user_course_meta
from mealrec.runs import RESULTS

OUT = RESULTS / "g3_compare.json"


def seed_mean(paths, key_options=("recall@20", "recall_at_20")) -> tuple[np.ndarray, np.ndarray]:
    zs = [np.load(p) for p in paths]
    key = next(k for k in key_options if k in zs[0].files)
    users = zs[0]["users"].astype(np.int64)
    assert all((z["users"].astype(np.int64) == users).all() for z in zs)
    return users, np.mean([z[key] for z in zs], axis=0)


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def main() -> None:
    rows = {}
    for cell in CELLS:
        name, protocol = cell.split("/")
        data = temporal(load(name), uc=user_course_meta(name)) if protocol == "p2" else get(protocol, name)
        nn_u, nn = seed_mean(sorted((RESULTS / "runs_b1n" / name / protocol / "full").glob("seed*.npz")))
        gb_dir = RESULTS / "runs_b1" / name / ("" if protocol == "p2" else protocol)
        gb_u, gb = seed_mean(sorted(gb_dir.glob("seed*.npz")))
        lookups = {b: baseline_per_user(data, b) for b in ("TrainCover", "CoverHeuristic")}
        best = max(lookups, key=lambda b: lookups[b][1].mean())
        lk_u, lk = lookups[best]
        assert (nn_u == gb_u).all() and (nn_u == lk_u).all()
        rows[cell] = {"CompMeal-N": float(nn.mean()), "GBDT": float(gb.mean()), "lookup": best,
                      "lookup_r20": float(lk.mean()), "n_users": int(len(nn_u)),
                      "p_vs_GBDT": float(wilcoxon(nn, gb).pvalue), "p_vs_lookup": float(wilcoxon(nn, lk).pvalue),
                      "p_GBDT_vs_lookup": float(wilcoxon(gb, lk).pvalue)}
    for k in ("p_vs_GBDT", "p_vs_lookup", "p_GBDT_vs_lookup"):
        for cell, p in zip(rows, holm([rows[c][k] for c in rows])):
            rows[cell][k + "_holm"] = p
    print(f"{'cell':<18}{'CompMeal-N':>11}{'GBDT':>8}{'lookup':>8}  p(N vs GBDT)  p(N vs lookup)  p(GBDT vs lookup)  [Holm]")
    for cell, r in rows.items():
        print(f"{cell:<18}{r['CompMeal-N']:>11.4f}{r['GBDT']:>8.4f}{r['lookup_r20']:>8.4f}  "
              f"{r['p_vs_GBDT_holm']:>12.2g}  {r['p_vs_lookup_holm']:>14.2g}  {r['p_GBDT_vs_lookup_holm']:>17.2g}  ({r['lookup']})")
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
