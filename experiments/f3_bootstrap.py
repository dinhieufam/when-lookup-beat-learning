"""F3: confidence intervals that include seed variance, for the paper's key comparisons.

The Wilcoxon tests in g5_summary / f1 treat each model as fixed: per-user Recall@20 is first
averaged over seeds. Here a two-level bootstrap resamples seeds (independently per model) and
users (shared, so the comparison stays paired) and reports the mean difference A - B with a 95%
percentile interval. An interval that excludes 0 is a difference robust to both sources of
noise; one that includes 0 is "no significant difference", not evidence of equality.

    python -m experiments.f3_bootstrap
"""
from __future__ import annotations

import json

import numpy as np

from mealrec.runs import RESULTS, per_user

OUT = RESULTS / "f3_bootstrap.json"
B = 2000
SEED = 0


def gnn(protocol: str, dataset: str, model: str, root: str = "runs") -> tuple[np.ndarray, np.ndarray]:
    runs = []
    for s in range(5):
        try:
            runs.append(per_user(protocol, dataset, model, s, root=root))
        except FileNotFoundError:
            pass
    users = runs[0]["users"]
    assert all((r["users"] == users).all() for r in runs)
    key = next(k for k in runs[0] if k.startswith("recall") and "20" in k)
    return users, np.stack([r[key] for r in runs])


def npz_runs(folder) -> tuple[np.ndarray, np.ndarray]:
    zs = [np.load(p) for p in sorted(folder.glob("seed*.npz"))]
    users = zs[0]["users"].astype(np.int64)
    key = next(k for k in ("recall@20", "recall_at_20") if k in zs[0].files)
    assert all((z["users"].astype(np.int64) == users).all() for z in zs)
    return users, np.stack([z[key] for z in zs])


def tuned_source(dataset: str, gnn_name: str) -> tuple[np.ndarray, np.ndarray]:
    """The GNN run used in Table 4 (as listed in g5_summary tests_vs_tuned)."""
    src = next(t["source"] for t in json.loads((RESULTS / "g5_summary.json").read_text())["tests_vs_tuned"]
               if t["dataset"] == dataset and t["gnn"] == gnn_name)
    root, model = src.split("/")
    return gnn("p1", dataset, model, root=root)


def boot(a: tuple, b: tuple, rng: np.random.Generator) -> dict:
    (ua, A), (ub, Bm) = a, b
    common = np.intersect1d(ua, ub)
    A = A[:, np.searchsorted(ua, common)] if not np.array_equal(ua, common) else A
    Bm = Bm[:, np.searchsorted(ub, common)] if not np.array_equal(ub, common) else Bm
    n = len(common)
    diffs = np.empty(B)
    for i in range(B):
        a_s = A[rng.integers(len(A), size=len(A))].mean(axis=0)
        b_s = Bm[rng.integers(len(Bm), size=len(Bm))].mean(axis=0)
        idx = rng.integers(n, size=n)
        diffs[i] = a_s[idx].mean() - b_s[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(A.mean() - Bm.mean()), "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0),
            "seeds": [int(len(A)), int(len(Bm))], "n_users": int(n)}


def comparisons() -> dict:
    c = {
        # iFashion: the order of MultiCBR and CrossCBR.
        "iFashion/released/MultiCBR-CrossCBR": (gnn("orig", "iFashion", "MultiCBR"), gnn("orig", "iFashion", "CrossCBR")),
        "iFashion/p1_upstream/CrossCBR-MultiCBR": (gnn("p1", "iFashion", "CrossCBR"), gnn("p1", "iFashion", "MultiCBR")),
        "iFashion/p1_tuned/MultiCBR-CrossCBR": (tuned_source("iFashion", "MultiCBR"), tuned_source("iFashion", "CrossCBR")),
    }
    # MealRec+: the cooperative advantage (vs BGCN) and the course-level advantage (vs LightGCN).
    for ds in ("MealRec+L", "MealRec+H"):
        for m in ("CrossCBR", "MultiCBR"):
            c[f"{ds}/released/{m}-BGCN"] = (gnn("orig", ds, m), gnn("orig", ds, "BGCN"))
            c[f"{ds}/p1_upstream/{m}-BGCN"] = (gnn("p1", ds, m), gnn("p1", ds, "BGCN"))
            c[f"{ds}/p1_tuned/{m}-BGCN"] = (tuned_source(ds, m), tuned_source(ds, "BGCN"))
        for m in ("BGCN", "CrossCBR", "MultiCBR"):
            c[f"{ds}/released/{m}-LightGCN"] = (gnn("orig", ds, m), gnn("orig", ds, "LightGCN"))
            c[f"{ds}/p1_upstream/{m}-LightGCN"] = (gnn("p1", ds, m), gnn("p1", ds, "LightGCN"))
            c[f"{ds}/p1_tuned/{m}-LightGCN"] = (tuned_source(ds, m), tuned_source(ds, "LightGCN"))
    # Table 4: BundleComp (and CompMeal) against each tuned GNN.
    gnns = {"MealRec+H": ("LightGCN", "BGCN", "CrossCBR", "MultiCBR"), "MealRec+L": ("LightGCN", "BGCN", "CrossCBR", "MultiCBR"),
            "Youshu": ("CrossCBR", "MultiCBR"), "NetEase": ("CrossCBR", "MultiCBR"), "iFashion": ("CrossCBR", "MultiCBR")}
    for ds, ms in gnns.items():
        bc = npz_runs(RESULTS / "runs_g5" / ds)
        for m in ms:
            c[f"{ds}/p1_tuned/BundleComp-{m}"] = (bc, tuned_source(ds, m))
    for ds in ("MealRec+H", "MealRec+L"):
        c[f"{ds}/p1_tuned/CompMeal-CrossCBR"] = (npz_runs(RESULTS / "runs_b1" / ds / "p1"), tuned_source(ds, "CrossCBR"))
    return c


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {}
    for label, (a, b) in comparisons().items():
        out[label] = boot(a, b, rng)
        r = out[label]
        print(f"{label:<45} diff {r['diff']:+.4f}  CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
              f"{'*' if r['excludes_zero'] else ' '}  seeds {r['seeds']}", flush=True)
    OUT.write_text(json.dumps({"B": B, "seed": SEED, "comparisons": out}, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
