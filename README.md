# Lookup probe for bundle recommendation benchmarks

Code for the paper *When Lookup Beats Learning: A Label Leak in Bundle Recommendation Benchmarks and How to Evaluate Without It* (anonymous submission).

## What is here

| Path | Contents |
|---|---|
| `mealrec/baselines.py` | The lookup probe (`cover_heuristic`), the legitimate lookups (`train_cover`, `most_pop`) |
| `mealrec/protocols.py` | Clean protocols: P1 (mask leaked user–item edges), P2 (temporal), cold-start splits |
| `mealrec/diagnostics.py` | Leak statistics (coverage of positives, held-out-only edges) |
| `mealrec/metrics.py` | Full-ranking Recall@K / NDCG@K |
| `experiments/` | One script per result; each writes a JSON to `results/` |
| `results/*.json` | Every number reported in the paper |

## Reproduce the leak in minutes (CPU)

```bash
pip install -r requirements.txt
bash scripts/download_mealrecplus.sh       # MealRec+ H and L
python -m experiments.d1_leakage           # MealRec+: lookup Recall@20 = 0.966 (H), 0.938 (L)
python -m experiments.g1_bundle_canary     # Youshu, NetEase, iFashion (CrossCBR release; ~8 min)
python -m pytest -q
```

The Youshu/NetEase/iFashion files are the CrossCBR release; place them under `data/raw/bundle_crosscbr/datasets/` (see `data/README.md`).

## Train the graph models (GPU)

```bash
bash scripts/setup_baselines.sh            # official CrossCBR, MultiCBR, BGCN/LightGCN, CoHeat code, pinned commits
python -m experiments.train_bundle --model CrossCBR --dataset MealRec+H --protocol p1 --seed 0
```

Job lists for every reported run are in `results/*_jobs.txt` (run with `scripts/run_queue.py`).

## The probe rule

Report every result next to two training-free numbers on the same protocol: the lookup probe (`cover_heuristic`) and the best legitimate lookup (the better of `most_pop` and `train_cover`, which read only training data). A protocol passes when the probe does not significantly exceed the best legitimate lookup (`experiments/f2_probe_rule.py`). Passing is necessary, not sufficient.
