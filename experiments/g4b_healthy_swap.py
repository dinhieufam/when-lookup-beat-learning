"""G4b: healthy swap with a bounded relevance loss, against hard filtering and re-weighting.

Backbone: CompMeal-N scores (results/runs_b1n_scores, saved by b1n_compmeal_nn --save-scores)
on P2, P1 and cold_p1. Healthiness uses scores recomputed from the released nutrition (D6):
WHO from energy shares (0..7, higher is healthier; the most reproducible variant) as the
target, FSA with per-serving amounts read as per 100 g (4..12, lower is healthier) reported
alongside. A meal's score is the mean of its courses'.

Each method turns the backbone into top-20 lists, with one knob swept:
- HardFilter(theta): only meals with WHO >= theta (the paper's FSA <= 8 filter, on WHO).
- ReWeight(lambda): per-user z-scored relevance + lambda * meal WHO (the paper's
  re-weighting, made scale-free).
- Swap(eps): keep the backbone list; replace each meal by its healthiest catalog neighbour
  that differs in exactly one course, is not a training meal of the user nor already in the
  list, is healthier, and whose relevance is at most eps * (user's score sd) lower.
The output is a Pareto curve of test R@20 against mean list WHO per method; the summary
reads off R@20 at fixed WHO gains by interpolation.

    python -m experiments.g4b_healthy_swap
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.d6_health_audit import course_scores, meal_mean
from mealrec.data import DATASETS, load
from mealrec.meta import user_course_meta
from mealrec.metrics import mask_scores, per_user_metrics, topk
from mealrec.protocols import get, temporal

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "g4b_healthy_swap.json"
SCORES = ROOT / "results" / "runs_b1n_scores"
VARIANT = "full__lr0.0003_wd0.0001"
PROTOCOLS = ("p2", "p1", "cold_p1")
K = 20
EPS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
GAINS = (0.25, 0.5, 1.0)


def neighbours(triples: np.ndarray) -> list[list[int]]:
    """Meals that share exactly two courses with each meal (differ in one slot)."""
    out = [set() for _ in range(len(triples))]
    for j in range(3):
        keep = [k for k in range(3) if k != j]
        groups = defaultdict(list)
        for m, t in enumerate(triples):
            groups[(t[keep[0]], t[keep[1]])].append(m)
        for ms in groups.values():
            for m in ms:
                out[m].update(x for x in ms if x != m)
    return [sorted(s) for s in out]


def meal_triples(data) -> np.ndarray:
    rows = data.meal_course.tolil().rows
    t = np.zeros((data.n_meals, 3), np.int64)
    for m, cs in enumerate(rows):
        for c in cs:
            t[m, data.course_category[c]] = c
    return t


def evaluate(lists: np.ndarray, users, test, who, fsa) -> dict:
    pu = per_user_metrics(lists, test[users], (K,))
    return {"recall@20": float(pu["recall@20"].mean()), "ndcg@20": float(pu["ndcg@20"].mean()),
            "who": float(who[lists].mean()), "fsa": float(fsa[lists].mean())}


def swap_lists(base: np.ndarray, s: np.ndarray, users, train, nbr, who, eps: float) -> tuple[np.ndarray, float]:
    out, swapped = base.copy(), 0
    for r, u in enumerate(users):
        sd = s[u][~np.isinf(s[u])].std()
        seen = set(base[r].tolist()) | set(train[u].indices.tolist())
        for i, m in enumerate(base[r]):
            cands = [n for n in nbr[m] if n not in seen and who[n] > who[m] and s[u, n] >= s[u, m] - eps * sd]
            if cands:
                best = max(cands, key=lambda n: (who[n], s[u, n]))
                out[r, i] = best
                seen.add(best)
                swapped += 1
    return out, swapped / base.size


def at_gain(curve: list[dict], base: dict, gain: float) -> float | None:
    """R@20 where the curve first reaches base WHO + gain (linear interpolation), if it does."""
    pts = sorted(((c["who"] - base["who"], c["recall@20"]) for c in curve))
    for (g0, r0), (g1, r1) in zip(pts, pts[1:]):
        if g0 <= gain <= g1:
            return r0 if g1 == g0 else r0 + (r1 - r0) * (gain - g0) / (g1 - g0)
    return None


def main() -> None:
    result = {}
    for name in DATASETS:
        base_data = load(name)
        cs, _ = course_scores(base_data, name)
        who = meal_mean(cs["who_energy_shares"], base_data)
        fsa = meal_mean(cs["fsa_serving_as_100g"], base_data)
        nbr = neighbours(meal_triples(base_data))
        n_nbr = np.array([len(x) for x in nbr])
        thetas = np.quantile(who, (0.0, 0.3, 0.5, 0.7, 0.8, 0.9))
        for protocol in PROTOCOLS:
            data = temporal(base_data, uc=user_course_meta(name)) if protocol == "p2" else get(protocol, name)
            train, test = data.user_meal["train"], data.user_meal["test"]
            users = np.flatnonzero(test.getnnz(axis=1))
            curves = defaultdict(list)
            for path in sorted((SCORES / name / protocol / VARIANT).glob("scores_seed*.npy")):
                s = mask_scores(np.load(path).astype(np.float64), train)
                base = topk(s[users], K)
                curves["Backbone"].append([evaluate(base, users, test, who, fsa)])
                curves["HardFilter"].append([
                    {"theta": float(t), **evaluate(topk(np.where(who[None] >= t, s[users], -np.inf), K), users, test, who, fsa)}
                    for t in thetas])
                fin = np.where(np.isinf(s), np.nan, s)
                z = (s - np.nanmean(fin, 1, keepdims=True)) / np.nanstd(fin, 1, keepdims=True)
                curves["ReWeight"].append([
                    {"lambda": lam, **evaluate(topk(z[users] + lam * who[None], K), users, test, who, fsa)} for lam in LAMBDAS])
                rows = []
                for eps in EPS:
                    lists, share = swap_lists(base, s, users, train, nbr, who, eps)
                    rows.append({"eps": eps, "share_swapped": share, **evaluate(lists, users, test, who, fsa)})
                curves["Swap"].append(rows)
            # Average each sweep point over seeds.
            mean = {}
            for method, per_seed in curves.items():
                mean[method] = [{k: float(np.mean([seed[i][k] for seed in per_seed])) for k in per_seed[0][i]}
                                for i in range(len(per_seed[0]))]
            bl = mean["Backbone"][0]
            summary = {m: {f"recall@20_at_who+{g}": at_gain(mean[m], bl, g) for g in GAINS}
                       for m in ("HardFilter", "ReWeight", "Swap")}
            result.setdefault(name, {})[protocol] = {"n_seeds": len(curves["Backbone"]), "curves": mean,
                                                     "summary": summary, "meals_with_neighbours": float((n_nbr > 0).mean())}
            print(f"\n{name} {protocol}: backbone R@20 {bl['recall@20']:.4f}  WHO {bl['who']:.3f}  FSA {bl['fsa']:.3f}"
                  f"  ({len(curves['Backbone'])} seeds)", flush=True)
            for m in ("HardFilter", "ReWeight", "Swap"):
                pts = "  ".join(f"({c['who'] - bl['who']:+.2f}, {c['recall@20']:.3f})" for c in mean[m])
                print(f"  {m:<10} (dWHO, R@20): {pts}", flush=True)
            print("  R@20 at WHO gain:", {m: {g: (None if v is None else round(v, 4)) for g, v in
                                               zip(GAINS, summary[m].values())} for m in summary}, flush=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
