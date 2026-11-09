"""R1: checks a reviewer will ask for before trusting the paper's claims.

a. Lookup vs best tuned model, per user: on P1/P2 the best tuned model (D3b, test
   R@20 averaged over its 5 seeds per user) against the best legitimate training-free
   baseline; paired Wilcoxon signed-rank test.
b. CompMeal (B1) vs the lookup on P2, per user (results/runs_b1/, 5 seeds averaged),
   and vs the best tuned model.
c. Cold start on three meal splits: CoHeat vs TrainCover on cold / cold_p1 with split
   seeds 0, 1, 2 (D8 used seed 0 only).
d. P2 cutoff at the 70th, 80th (main) and 90th percentile: baselines and the two
   strongest GNNs (default settings, 3-5 seeds).

    python -m experiments.r1_reviewer_checks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from mealrec.baselines import BASELINES
from mealrec.data import DATASETS
from mealrec.metrics import mask_scores, per_user_metrics, topk
from mealrec.protocols import get
from mealrec.runs import RESULTS, per_user, summaries

OUT = RESULTS / "r1_reviewer_checks.json"


def baseline_per_user(data, name: str) -> tuple[np.ndarray, np.ndarray]:
    train, test = data.user_meal["train"], data.user_meal["test"]
    users = np.flatnonzero(test.getnnz(axis=1))
    top = topk(mask_scores(BASELINES[name](data)[users], train[users]), 80)
    return users, per_user_metrics(top, test[users])["recall@20"]


def paired(a_users, a, b_users, b) -> dict:
    assert (a_users == b_users).all()
    diff = a - b
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(diff.mean()),
            "share_users_a_better": float((diff > 0).mean()), "share_users_b_better": float((diff < 0).mean()),
            "wilcoxon_p": float(wilcoxon(a, b).pvalue) if np.any(diff) else 1.0, "n_users": int(len(diff))}


def seed_mean(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    users = rows[0]["users"]
    assert all((r["users"] == users).all() for r in rows)
    return users, np.mean([r["recall@20"] for r in rows], axis=0)


def check_a(d3b: dict) -> dict:
    tuned = summaries(root="runs_tuned")
    out = {}
    for dataset in DATASETS:
        for protocol in ("p1", "p2"):
            cell = d3b[dataset][protocol]
            model = cell["best_model"]
            tag = cell["models"][model]["setting"]
            seeds = sorted(tuned.query("protocol == @protocol and dataset == @dataset and model == @model "
                                       "and tag == @tag")["seed"])
            mu, m = seed_mean([per_user(protocol, dataset, f"{model}__{tag}", s, root="runs_tuned") for s in seeds])
            base = cell["best_legit_baseline"]
            bu, b = baseline_per_user(get(protocol, dataset), base)
            out.setdefault(dataset, {})[protocol] = {"a": base, "b": f"{model} (tuned, {len(seeds)} seeds)",
                                                     **paired(bu, b, mu, m)}
    return out


def check_b(d3b: dict) -> dict:
    out = {}
    for dataset in DATASETS:
        data = get("p2", dataset)
        files = sorted((RESULTS / "runs_b1" / dataset).glob("seed*.npz"))
        cu, c = seed_mean([dict(np.load(f)) for f in files])
        bu, b = baseline_per_user(data, "CoverHeuristic")
        cell = d3b[dataset]["p2"]
        model, tag = cell["best_model"], cell["models"][cell["best_model"]]["setting"]
        tuned = summaries(root="runs_tuned").query("protocol == 'p2' and dataset == @dataset and model == @model "
                                                   "and tag == @tag")
        mu, m = seed_mean([per_user("p2", dataset, f"{model}__{tag}", s, root="runs_tuned")
                           for s in sorted(tuned["seed"])])
        out[dataset] = {"vs_CoverHeuristic": {"a": f"CompMeal ({len(files)} seeds)", "b": "CoverHeuristic",
                                              **paired(cu, c, bu, b)},
                        "vs_best_tuned_model": {"a": f"CompMeal ({len(files)} seeds)", "b": f"{model} (tuned)",
                                                **paired(cu, c, mu, m)}}
    return out


def check_c() -> dict:
    runs = summaries()
    out = {}
    for dataset in DATASETS:
        for base in ("cold", "cold_p1"):
            for split_seed in (0, 1, 2):
                protocol = base if split_seed == 0 else f"{base}_s{split_seed}"
                r = runs.query("protocol == @protocol and dataset == @dataset and model == 'CoHeat'")
                if r.empty:
                    continue
                data = get(protocol, dataset)
                tc = float(baseline_per_user(data, "TrainCover")[1].mean())
                cover = float(baseline_per_user(data, "CoverHeuristic")[1].mean())
                out.setdefault(dataset, {}).setdefault(base, {})[f"split{split_seed}"] = {
                    "CoHeat_mean": float(r["tune_sel/recall@20"].mean()),
                    "CoHeat_sd": float(r["tune_sel/recall@20"].std(ddof=1)), "n_seeds": int(len(r)),
                    "TrainCover": tc, "CoverHeuristic": cover}
    return out


def check_d() -> dict:
    runs = summaries()
    out = {}
    for dataset in DATASETS:
        for protocol in ("p2_q70", "p2", "p2_q90"):
            data = get(protocol, dataset)
            row = {"pairs": {s: int(m.nnz) for s, m in data.user_meal.items()}, "cutoffs": data.cutoffs,
                   "baselines": {b: float(baseline_per_user(data, b)[1].mean()) for b in BASELINES}}
            for model in ("MultiCBR", "CrossCBR"):
                r = runs.query("protocol == @protocol and dataset == @dataset and model == @model")
                row[model] = {"mean": float(r["tune_sel/recall@20"].mean()), "n_seeds": int(len(r))}
            legit = {k: v for k, v in row["baselines"].items()}  # on P2 every baseline reads only pre-cutoff data
            row["best_legit_baseline"] = max(legit, key=legit.get)
            row["best_model"] = max(("MultiCBR", "CrossCBR"), key=lambda m: row[m]["mean"])
            row["lookup_beats_models"] = legit[row["best_legit_baseline"]] > row[row["best_model"]]["mean"]
            out.setdefault(dataset, {})[protocol] = row
    return out


def main() -> None:
    d3b = json.loads((RESULTS / "d3b_tuning.json").read_text())
    result = {"a_lookup_vs_tuned": check_a(d3b), "b_compmeal": check_b(d3b),
              "c_cold_split_seeds": check_c(), "d_p2_cutoff": check_d()}
    for key, block in result.items():
        print(f"\n== {key}")
        print(json.dumps(block, indent=1, default=float)[:3000])
    OUT.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nwrote {OUT.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
