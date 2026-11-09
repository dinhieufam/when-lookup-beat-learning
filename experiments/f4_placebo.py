"""F4: a placebo for the counterfactual of Table 3.

The counterfactual keeps the weights of a model trained on the released data and removes the
leaked user-item edges (those P1 drops) at inference. Part of the drop could come from removing
edges at all (a distribution shift), not from the leak. The placebo removes, for every user, the
same number of edges drawn uniformly from the user's edges that P1 keeps (non-leaked edges), and
re-scores with the same weights. If the counterfactual drop is about the leak, the placebo drop
should be much smaller.

For every (dataset, model, seed) run on the released data (results/runs/orig, upstream settings):
released, counterfactual and placebo test Recall@20, with per-user arrays in results/runs_f4/.
GPU, one process; ~20 minutes.

    python -m experiments.f4_placebo
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import scipy.sparse as sp
import torch

from experiments.train_bundle import build, chunked_top, evaluate_top, parse
from mealrec.protocols import get
from mealrec.runs import RESULTS

OUT = RESULTS / "f4_placebo.json"
PER_USER = RESULTS / "runs_f4"
RUNS = {"iFashion": ("MultiCBR", "CrossCBR"),
        "MealRec+L": ("CrossCBR", "MultiCBR", "BGCN"),
        "MealRec+H": ("CrossCBR", "MultiCBR", "BGCN")}
SEEDS = range(5)


def placebo_graph(orig: sp.csr_matrix, p1: sp.csr_matrix, rng: np.random.Generator) -> tuple[sp.csr_matrix, dict]:
    """Remove, per user, as many non-leaked edges as P1 removes leaked ones (capped at what the user has)."""
    orig, p1 = orig.tocsr(), p1.tocsr()
    n_drop = orig.getnnz(axis=1) - p1.getnnz(axis=1)
    keep = p1.tolil(copy=True)
    removed = 0
    for u in np.flatnonzero(n_drop > 0):
        cols = np.asarray(keep.rows[u])
        k = min(int(n_drop[u]), len(cols))
        gone = set(rng.choice(cols, size=k, replace=False).tolist())
        keep.rows[u] = [c for c in keep.rows[u] if c not in gone]
        keep.data[u] = [1.0] * len(keep.rows[u])
        removed += k
    # keep holds the P1 graph minus the placebo edges; add the leaked edges back.
    leaked = orig - orig.multiply(p1)
    out = sp.csr_matrix(keep.tocsr() + leaked, dtype=np.float32)
    out.data[:] = 1.0
    return out, {"leaked_edges": int(n_drop.sum()), "placebo_edges": int(removed)}


@torch.no_grad()
def test_recall(model: str, data, run_args: dict, state: dict, device) -> tuple[np.ndarray, np.ndarray]:
    argv = ["--model", model, "--dataset", data.name.split("/")[0], "--seed", str(run_args["seed"]),
            "--dim", str(run_args["dim"]), "--layers", str(run_args["layers"])]
    if run_args.get("lr"):
        argv += ["--lr", str(run_args["lr"])]
    if run_args.get("l2") is not None:
        argv += ["--l2", str(run_args["l2"])]
    ad = build(model, data, parse(argv), device)
    ad.model.load_state_dict(state, strict=False)  # graph tensors come from `data`, weights from the run
    ev = evaluate_top(chunked_top(ad, data, device), data)["test"]
    return ev["users"], ev["recall@20"]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PER_USER.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for dataset, models in RUNS.items():
        orig, p1 = get("orig", dataset), get("p1", dataset)
        placebo = {}
        for seed in SEEDS:  # one placebo draw per seed, shared by the models
            g, info = placebo_graph(orig.user_course, p1.user_course, np.random.default_rng(1000 + seed))
            placebo[seed] = (replace(orig, name=f"{dataset}/placebo", user_course=g), info)
        for model in models:
            key = f"{dataset}/{model}"
            if key in results:
                print(f"done already: {key}")
                continue
            rows = []
            for seed in SEEDS:
                run = RESULTS / "runs" / "orig" / dataset / model / f"seed{seed}"
                run_args = json.loads((run / "summary.json").read_text())["args"]
                state = torch.load(run / "model_tune_best.pt", map_location=device)
                row = {"seed": seed, **placebo[seed][1]}
                for variant, data in (("released", orig), ("counterfactual", p1), ("placebo", placebo[seed][0])):
                    users, r = test_recall(model, data, run_args, state, device)
                    np.savez_compressed(PER_USER / f"{dataset}__{model}__{variant}__seed{seed}.npz", users=users, recall=r)
                    row[variant] = float(r.mean())
                rows.append(row)
                print(f"{key} seed {seed}: released {row['released']:.4f} counterfactual {row['counterfactual']:.4f} "
                      f"placebo {row['placebo']:.4f} ({row['placebo_edges']}/{row['leaked_edges']} edges)", flush=True)
                del state
                torch.cuda.empty_cache()
            mean = {v: float(np.mean([r[v] for r in rows])) for v in ("released", "counterfactual", "placebo")}
            results[key] = {"seeds": rows, "mean": mean,
                            "drop_counterfactual": 1 - mean["counterfactual"] / mean["released"],
                            "drop_placebo": 1 - mean["placebo"] / mean["released"]}
            OUT.write_text(json.dumps(results, indent=2) + "\n")
            print(f"{key}: counterfactual -{results[key]['drop_counterfactual']:.1%}, "
                  f"placebo -{results[key]['drop_placebo']:.1%}", flush=True)
    print(f"wrote {OUT.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
