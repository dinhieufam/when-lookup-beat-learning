"""F6: BundleComp without its visible-coverage feature (cover_ui), under P1.

Under P1 a held-out positive's visible items are all explained by the user's training bundles, and
replay teaches the ranker which visible-coverage patterns occur at test time. If that is what
gives BundleComp its lead, removing cover_ui should remove the lead. Compared with the full
BundleComp and with the best tuned graph model of Table 5 (bootstrap over seeds and users, as f3).

    python -m experiments.g5_bundlecomp --drop cover_ui --tag nocover   # the runs
    python -m experiments.f6_bundlecomp_ablation

CPU, ~1 minute.
"""
from __future__ import annotations

import json

import numpy as np

from experiments.f3_bootstrap import B, RESULTS, boot, npz_runs, tuned_source

OUT = RESULTS / "f6_bundlecomp_ablation.json"
# iFashion is left out: BundleComp already trails the graph models there, so the ablation cannot change a conclusion.
DATASETS = ("MealRec+H", "MealRec+L", "Youshu", "NetEase")


def main() -> None:
    rng = np.random.default_rng(0)
    tuned = {(r["dataset"], r["gnn"]): r["gnn_r20"]
             for r in json.loads((RESULTS / "g5_summary.json").read_text())["tests_vs_tuned"]}
    out = {}
    for ds in DATASETS:
        full, abl = npz_runs(RESULTS / "runs_g5" / ds), npz_runs(RESULTS / "runs_g5_nocover" / ds)
        best = max(("CrossCBR", "MultiCBR"), key=lambda m: tuned[(ds, m)])
        gnn = tuned_source(ds, best)
        out[ds] = {"full": float(full[1].mean()), "nocover": float(abl[1].mean()), "seeds": [len(full[1]), len(abl[1])],
                   "best_gnn": best, "best_gnn_r20": float(gnn[1].mean()),
                   "nocover_minus_full": boot(abl, full, rng), "nocover_minus_best_gnn": boot(abl, gnn, rng)}
        r = out[ds]
        print(f"{ds:<10} full {r['full']:.4f}  no cover_ui {r['nocover']:.4f}  "
              f"diff {r['nocover_minus_full']['diff']:+.4f} {np.round(r['nocover_minus_full']['ci95'], 4)}  "
              f"vs {best} {r['best_gnn_r20']:.4f}: {r['nocover_minus_best_gnn']['diff']:+.4f} "
              f"{np.round(r['nocover_minus_best_gnn']['ci95'], 4)}", flush=True)
    OUT.write_text(json.dumps({"B": B, "seed": 0, "datasets": out}, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
