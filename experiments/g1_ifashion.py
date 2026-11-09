"""G1: on iFashion, how much do published-style models owe to the unsplit user_item.txt?

iFashion ships an unsplit user_item.txt; a training-free lookup over it beats the published
CrossCBR and MultiCBR numbers (g1_bundle_canary). Here, with the upstream iFashion configs
and 5 seeds (results/g1_ifashion_jobs.txt):
1. orig            trained and evaluated on the released data;
2. counterfactual  orig weights, P1 graph at inference (user_item edges explained only by
                   the user's held-out bundles removed), as in D4;
3. p1              retrained and evaluated on P1.
LightGCN reads no user_item, so it is the control (orig = P1 by construction).

    python scripts/run_queue.py --jobs-file results/g1_ifashion_jobs.txt
    python -m experiments.g1_ifashion
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.train_bundle import build, chunked_top, evaluate_top, parse
from mealrec.protocols import get
from mealrec.runs import run_dir, summaries

OUT = Path(__file__).resolve().parent.parent / "results" / "g1_ifashion.json"
DATASET = "iFashion"
MODELS = ("LightGCN", "CrossCBR", "MultiCBR")
METRICS = ("recall@20", "ndcg@20")


def counterfactual(model: str, seed: int, p1, device) -> dict:
    a = parse(["--model", model, "--dataset", DATASET, "--seed", str(seed)])
    ad = build(model, p1, a, device)
    state = torch.load(run_dir("orig", DATASET, model, seed) / "model_tune_best.pt", map_location=device)
    ad.model.load_state_dict(state, strict=False)  # graph tensors are rebuilt from p1, weights reused
    test = evaluate_top(chunked_top(ad, p1, device), p1)["test"]
    return {m: float(test[m].mean()) for m in METRICS}


def stats(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = summaries().query("dataset == @DATASET")
    p1 = get("p1", DATASET)
    result = {}
    for model in MODELS:
        row = {}
        for protocol in ("orig", "p1"):
            r = runs.query("model == @model and protocol == @protocol")
            if len(r):
                row[protocol] = {m: stats(r[f"tune_sel/{m}"].tolist()) for m in METRICS}
        seeds = sorted(runs.query("model == @model and protocol == 'orig'")["seed"])
        if model != "LightGCN" and seeds:
            cf = [counterfactual(model, int(s), p1, device) for s in seeds]
            row["counterfactual"] = {m: stats([c[m] for c in cf]) for m in METRICS}
            row["relative_drop"] = {
                k: 1 - row[k]["recall@20"]["mean"] / row["orig"]["recall@20"]["mean"]
                for k in ("counterfactual", "p1") if k in row}
        result[model] = row
        line = " | ".join(f"{k} {v['recall@20']['mean']:.4f}±{v['recall@20']['sd']:.4f}"
                          for k, v in row.items() if k != "relative_drop")
        print(f"{model:<9} R@20: {line}  drop {row.get('relative_drop', {})}", flush=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
