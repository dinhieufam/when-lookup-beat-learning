"""F5: graph models trained by protocol replay (results/runs_replay, --replay 5) vs. the same
models trained normally on P1 (Table 5 settings), and the simple rankers vs. the best replay model.

Bootstrap over seeds and users as in f3. CPU, ~2 minutes.

    python -m experiments.f5_replay
"""
from __future__ import annotations

import json

import numpy as np

from experiments.f3_bootstrap import B, RESULTS, boot, gnn, npz_runs, tuned_source

OUT = RESULTS / "f5_replay.json"
MODELS = {"MealRec+H": ("CrossCBR", "MultiCBR", "BGCN"), "MealRec+L": ("CrossCBR", "MultiCBR", "BGCN"),
          "Youshu": ("CrossCBR", "MultiCBR"), "NetEase": ("CrossCBR", "MultiCBR"), "iFashion": ("CrossCBR", "MultiCBR")}


def replay_runs(dataset: str, model: str):
    folder = next((RESULTS / "runs_replay" / "p1" / dataset).glob(f"{model}__*"))
    return gnn("p1", dataset, folder.name, root="runs_replay")


def main() -> None:
    rng = np.random.default_rng(0)
    out = {}
    for ds, models in MODELS.items():
        row = {"models": {}}
        for m in models:
            rep, nor = replay_runs(ds, m), tuned_source(ds, m)
            row["models"][m] = {"replay": float(rep[1].mean()), "normal": float(nor[1].mean()),
                                "replay_minus_normal": boot(rep, nor, rng)}
        best = max(row["models"], key=lambda m: row["models"][m]["replay"])
        row["best_replay"] = best
        rep = replay_runs(ds, best)
        row["BundleComp_minus_best_replay"] = boot(npz_runs(RESULTS / "runs_g5" / ds), rep, rng)
        if ds.startswith("MealRec"):
            row["CompMeal_minus_best_replay"] = boot(npz_runs(RESULTS / "runs_b1" / ds / "p1"), rep, rng)
        out[ds] = row
        print(ds, {m: (round(v["replay"], 4), round(v["normal"], 4), [round(x, 4) for x in v["replay_minus_normal"]["ci95"]])
                   for m, v in row["models"].items()}, "best", best,
              "BC-best", round(row["BundleComp_minus_best_replay"]["diff"], 4),
              [round(x, 4) for x in row["BundleComp_minus_best_replay"]["ci95"]], flush=True)
    OUT.write_text(json.dumps({"B": B, "seed": 0, "datasets": out}, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
