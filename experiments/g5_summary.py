"""G5 summary: the paper's main table across the five datasets, with Holm-corrected tests.

Per dataset (test R@20):
- lookups: CoverHeuristic on the released data (the leak canary), CoverHeuristic under P1 and
  TrainCover (legitimate) from results/g1_bundle_canary.json;
- GNNs (5 seeds, epoch by tune) on the released split and under P1, from results/runs/;
- BundleComp (results/runs_g5/, g5_bundlecomp) and, for meal datasets, the slot-wise CompMeal
  (results/runs_b1/<dataset>/p1/, b1_compmeal_static), both under P1.
BundleComp is compared per user with every GNN under P1 (Wilcoxon on seed-averaged R@20),
Holm-corrected over all (dataset, GNN) pairs, twice: against the GNNs as tuned on P1 (main
comparison: D3b settings for MealRec+, g5_tuning settings for the bundle datasets; HyperMBR
was not tuned and keeps its default) and against the upstream defaults.

    python -m experiments.g5_summary
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import wilcoxon

from experiments.g3_compare import holm, seed_mean
from mealrec.data import BUNDLE_DATASETS, DATASETS
from mealrec.runs import RESULTS, per_user, summaries

OUT = RESULTS / "g5_summary.json"
GNNS = ("LightGCN", "BGCN", "CrossCBR", "MultiCBR", "HyperMBR")


def tuned_source(name: str, model: str) -> tuple[str, str]:
    """(root, model directory) of the P1-tuned runs of a model; the default runs if it was not tuned."""
    if name in DATASETS:
        d3b = json.loads((RESULTS / "d3b_tuning.json").read_text())[name]["p1"]["models"]
        if model in d3b:
            return "runs_tuned", f"{model}__{d3b[model]['setting']}"
    else:
        g5 = json.loads((RESULTS / "g5_tuning.json").read_text())[name]
        if model in g5 and not g5[model]["setting"]["is_default"]:
            st = g5[model]["setting"]
            return "runs_tuned", f"{model}__lr{st['lr']:g}_l2{st['l2']:g}"
    return "runs", model


def compare(name, model, root, model_dir, users, ours) -> dict:
    seeds = sorted(int(x.parent.name[4:]) for x in (RESULTS / root / "p1" / name / model_dir).glob("seed*/summary.json"))
    pus = [per_user("p1", name, model_dir, s, root=root) for s in seeds]
    assert pus and all((x["users"] == users).all() for x in pus)
    other = np.mean([x["recall@20"] for x in pus], axis=0)
    return {"dataset": name, "gnn": model, "source": f"{root}/{model_dir}", "n_seeds": len(seeds),
            "bundlecomp": float(ours.mean()), "gnn_r20": float(other.mean()), "diff": float((ours - other).mean()),
            "p": float(wilcoxon(ours, other).pvalue)}


def main() -> None:
    canary = json.loads((RESULTS / "g1_bundle_canary.json").read_text())
    runs = summaries(root="runs")
    table, tests, tuned_tests = {}, [], []
    for name in DATASETS + BUNDLE_DATASETS:
        c = canary[name]["baselines"]
        row = {"canary_orig": c["CoverHeuristic"]["recall@20"], "canary_p1": c["CoverHeuristic_p1"]["recall@20"],
               "train_cover": c["TrainCover"]["recall@20"], "gnn": {}}
        for model in GNNS:
            for protocol in ("orig", "p1"):
                r = runs.query("dataset == @name and model == @model and protocol == @protocol")
                if len(r):
                    row["gnn"].setdefault(model, {})[protocol] = {
                        "mean": float(r["tune_sel/recall@20"].mean()), "sd": float(r["tune_sel/recall@20"].std(ddof=1)),
                        "n": int(len(r))}
        users, ours = seed_mean(sorted((RESULTS / "runs_g5" / name).glob("seed*.npz")))
        row["bundlecomp_p1"] = float(ours.mean())
        if name in DATASETS:
            _, cm = seed_mean(sorted((RESULTS / "runs_b1" / name / "p1").glob("seed*.npz")))
            row["compmeal_p1"] = float(cm.mean())
        for model, d in row["gnn"].items():
            if "p1" not in d:
                continue
            tests.append(compare(name, model, "runs", model, users, ours))
            root, model_dir = tuned_source(name, model)
            tuned_tests.append(compare(name, model, root, model_dir, users, ours))
        row["best_gnn_p1_tuned"] = max(((t["gnn"], t["gnn_r20"]) for t in tuned_tests if t["dataset"] == name),
                                       key=lambda x: x[1])
        table[name] = row
    for group in (tests, tuned_tests):
        for t, p in zip(group, holm([t["p"] for t in group])):
            t["p_holm"] = p
    OUT.write_text(json.dumps({"table": table, "tests_vs_tuned": tuned_tests, "tests_vs_default": tests}, indent=2) + "\n")

    print("| Dataset | Canary (orig) | Canary (P1) | TrainCover | best GNN orig | best GNN P1 (default) | best GNN P1 (tuned) | CompMeal P1 | BundleComp P1 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in table.items():
        best = lambda prot: max(((m, d[prot]["mean"]) for m, d in r["gnn"].items() if prot in d), key=lambda x: x[1])
        (mo, vo), (mp, vp) = best("orig"), best("p1")
        print(f"| {name} | {r['canary_orig']:.3f} | {r['canary_p1']:.3f} | {r['train_cover']:.3f} | {vo:.3f} ({mo}) | "
              f"{vp:.3f} ({mp}) | {r['best_gnn_p1_tuned'][1]:.3f} ({r['best_gnn_p1_tuned'][0]}) | "
              f"{r.get('compmeal_p1', float('nan')):.3f} | {r['bundlecomp_p1']:.3f} |")
    print("\nBundleComp vs P1-tuned GNNs (Wilcoxon per user, Holm over all pairs):")
    for t in tuned_tests:
        print(f"  {t['dataset']:<10} {t['gnn']:<9} {t['gnn_r20']:.4f} -> {t['bundlecomp']:.4f}  diff {t['diff']:+.4f}  "
              f"p_holm {t['p_holm']:.2g}")
    print(f"wrote {OUT.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
