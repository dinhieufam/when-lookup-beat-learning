"""F1: derived numbers for the FrontierIR paper that no earlier result file states directly.

The workshop version leaves HyperMBR out (it does not reproduce), so every
statistic that pooled it is recomputed here over the remaining models:
- Holm correction of BundleComp vs. the tuned GNNs (g5_summary tests) without the HyperMBR rows;
- the range of test-set epoch-selection inflation (d2_epoch_selection) over the four other models.
It also adds two iFashion tests that the draft quoted from notes only:
- CrossCBR vs. MultiCBR retrained on P1 with upstream settings (paired Wilcoxon over users);
- the same pair re-tuned on P1 (g5_tuning settings);
the seed standard deviations behind Table 3 (released, counterfactual and P1 cells); and, on
MealRec+, the share of test positives whose three courses all lie in the user's training meals
(why the legitimate TrainCover lookup is already strong).
CPU only, a few seconds.

    python -m experiments.f1_paper_extras
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import wilcoxon

from mealrec.data import load as load_data
from mealrec.runs import RESULTS, per_user

OUT = RESULTS / "f1_paper_extras.json"
MODELS4 = ("LightGCN", "BGCN", "CrossCBR", "MultiCBR")
TUNED_IFASHION = {"CrossCBR": "CrossCBR__lr0.0001_l24e-06", "MultiCBR": "MultiCBR__lr0.0001_l21e-06"}


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjustment (same as experiments.g3_compare.holm, which imports torch)."""
    order, adj, running = np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return list(adj)


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def seed_avg_recall(protocol: str, dataset: str, model: str, root: str = "runs") -> tuple[np.ndarray, np.ndarray]:
    runs = [per_user(protocol, dataset, model, s, root=root) for s in range(5)]
    users = runs[0]["users"]
    assert all((r["users"] == users).all() for r in runs)
    key = next(k for k in runs[0] if k.startswith("recall") and "20" in k)
    return users, np.mean([r[key] for r in runs], axis=0)


def ifashion_pair(protocol: str, models: dict[str, str], root: str) -> dict:
    (u1, a), (u2, b) = (seed_avg_recall(protocol, "iFashion", models[m], root) for m in ("CrossCBR", "MultiCBR"))
    assert (u1 == u2).all()
    return {"CrossCBR": float(a.mean()), "MultiCBR": float(b.mean()), "n_users": int(len(u1)),
            "wilcoxon_p": float(wilcoxon(a, b).pvalue)}


def main() -> None:
    out: dict = {}

    # 1. Holm over BundleComp-vs-tuned-GNN tests without HyperMBR.
    tests = [t for t in load("g5_summary")["tests_vs_tuned"] if t["gnn"] != "HyperMBR"]
    adj = holm([t["p"] for t in tests])
    out["bundlecomp_vs_tuned_holm_no_hypermbr"] = {
        f"{t['dataset']}/{t['gnn']}": {"gnn_r20": t["gnn_r20"], "bundlecomp": t["bundlecomp"], "p": t["p"],
                                       "p_holm": float(h)} for t, h in zip(tests, adj)}

    # 2. Epoch-selection inflation over four models.
    d2 = load("d2_epoch_selection")
    out["epoch_selection_inflation_4models"] = {
        ds: {m: d2[ds]["models"][m]["inflation/recall@20/mean"] for m in MODELS4} for ds in ("MealRec+H", "MealRec+L")}

    # 3. iFashion: CrossCBR vs MultiCBR on P1, upstream settings and re-tuned.
    out["ifashion_p1_upstream"] = ifashion_pair("p1", {m: m for m in TUNED_IFASHION}, "runs")
    out["ifashion_p1_tuned"] = ifashion_pair("p1", TUNED_IFASHION, "runs_tuned")

    # 4. Seed SDs behind Table 3.
    s, d4, g1 = load("g5_summary")["table"], load("d4_shortcut"), load("g1_ifashion")
    sd = {}
    for ds, models in (("MealRec+H", ("BGCN", "CrossCBR", "MultiCBR")), ("MealRec+L", ("BGCN", "CrossCBR", "MultiCBR"))):
        for m in models:
            cf = [x["recall@20_p1_graph"] for x in d4[ds][m]["seeds"]]
            sd[f"{ds}/{m}"] = {"released": s[ds]["gnn"][m]["orig"]["sd"], "counterfactual": float(np.std(cf, ddof=1)),
                               "p1": s[ds]["gnn"][m]["p1"]["sd"]}
    for m in ("CrossCBR", "MultiCBR"):
        sd[f"iFashion/{m}"] = {k: g1[m][v]["recall@20"]["sd"] for k, v in
                               (("released", "orig"), ("counterfactual", "counterfactual"), ("p1", "p1"))}
    out["table3_sd"] = sd
    flat = {f"{k}/{c}": v for k, cells in sd.items() for c, v in cells.items()}
    worst = max(flat, key=flat.get)
    out["table3_sd_max"] = {"cell": worst, "sd": flat[worst],
                            "max_excluding_mealrecL_bgcn": max(v for k, v in flat.items() if not k.startswith("MealRec+L/BGCN"))}

    # 5. Test positives fully covered by the user's training meals (MealRec+).
    out["test_positive_covered_by_train_meals"] = {}
    for ds in ("MealRec+H", "MealRec+L"):
        d = load_data(ds)
        train_courses = ((d.user_meal["train"] @ d.meal_course) > 0).astype(np.float32)  # U x C
        test = d.user_meal["test"].tocoo()
        hits = np.asarray((train_courses[test.row].multiply(d.meal_course[test.col])).sum(axis=1)).ravel()
        out["test_positive_covered_by_train_meals"][ds] = float((hits >= 3).mean())

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("ifashion_p1_upstream", "ifashion_p1_tuned", "table3_sd_max")}, indent=1))
    print(f"wrote {OUT.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
