"""D6 (full): are the MealRec+ healthiness scores and conclusions reproducible and robust?

1. Reproducing course scores from the released per-serving nutrition (course.csv):
   - FSA traffic light (UK FSA front-of-pack, thresholds per 100 g) needs the serving
     mass, which is not released. Variants: per-serving amounts read as per 100 g, and
     every serving assumed to weigh k grams, with k fitted to the released labels
     (an optimistic upper bound on reproducibility).
   - WHO (Howard et al. 2012; Trattner & Elsweiler 2017): 7 criteria on energy shares
     plus fibre and sodium per 2000 kcal. It needs no mass.
2. Do the paper's conclusions survive?
   a. "Less healthy phenomenon": interacted meals are less healthy than the catalogue,
      under released and recomputed scores.
   b. ExpoGap of trained models (results/runs, all protocols) and baselines: the
      healthier half is cut at the median of a discrete score, and 15% of meals tie at
      the cut. Upstream breaks ties with an unstable argsort. We report the stable
      index order, the range over 100 random tie orders, and a strict split that drops
      tied meals.
   c. Delta FSA against the user's history: the released user_fsa is the mean over the
      user's training meals of the ORIGINAL split, so under P2 it contains meals after the
      cutoff; we also recompute it from each protocol's own training meals.

    python -m experiments.d6_health_audit
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from mealrec.baselines import BASELINES
from mealrec.data import DATASETS
from mealrec.meta import course_meta, nutrient
from mealrec.metrics import mask_scores, topk
from mealrec.protocols import PROTOCOLS, get
from mealrec.runs import MODELS, scores, summaries

OUT = Path(__file__).resolve().parent.parent / "results" / "d6_health_audit.json"
FSA_100G = {"fat": (3, 17.5), "saturatedFat": (1.5, 5), "sugars": (5, 22.5), "salt": (0.3, 1.5)}
K = 20
N_TIE_DRAWS = 100


def fsa_points(per100: dict[str, np.ndarray]) -> np.ndarray:
    pts = 0
    for key, (green, red) in FSA_100G.items():
        x = per100[key]
        pts = pts + np.where(x <= green, 1, np.where(x > red, 3, 2))
    return pts


def who_score(n: dict[str, np.ndarray]) -> np.ndarray:
    kcal = np.where(n["calories"] > 0, n["calories"], np.nan)
    share = lambda grams, kcal_per_g: 100 * grams * kcal_per_g / kcal
    crit = [
        (share(n["protein"], 4) >= 10) & (share(n["protein"], 4) <= 15),
        (share(n["carbohydrates"], 4) >= 55) & (share(n["carbohydrates"], 4) <= 75),
        share(n["sugars"], 4) < 10,
        (share(n["fat"], 9) >= 15) & (share(n["fat"], 9) <= 30),
        share(n["saturatedFat"], 9) < 10,
        n["fiber"] / kcal * 2000 > 25,
        n["sodium"] / 1000 / kcal * 2000 < 2,
    ]
    return np.sum(crit, axis=0).astype(float)


def agreement(recomputed: np.ndarray, released: np.ndarray) -> dict:
    return {"exact": float(np.mean(recomputed == released)),
            "within_1": float(np.mean(np.abs(recomputed - released) <= 1)),
            "spearman": float(spearmanr(recomputed, released).statistic)}


def course_scores(data, name) -> tuple[dict, dict]:
    """Recomputed course-level scores (indexed like data.course_fsa) and their agreement."""
    meta = course_meta(name, columns=("nutritions",))
    idx = meta.index.to_numpy()
    n = {k: np.nan_to_num(nutrient(meta["nutritions"], k)) for k in
         ("fat", "saturatedFat", "sugars", "sodium", "calories", "protein", "carbohydrates", "fiber")}
    n["salt"] = n["sodium"] * 2.5 / 1000
    used = np.isin(idx, np.flatnonzero(data.meal_course.getnnz(axis=0)))
    released_fsa, released_who = data.course_fsa[idx], data.course_who[idx]

    variants = {"fsa_serving_as_100g": fsa_points({k: n[k] for k in FSA_100G})}
    fit = {}
    for k in range(50, 1001, 10):
        pts = fsa_points({key: n[key] * 100 / k for key in FSA_100G})
        fit[k] = float(np.mean(pts[used] == released_fsa[used]))
    best_k = max(fit, key=fit.get)
    variants[f"fsa_constant_mass_{best_k}g"] = fsa_points({key: n[key] * 100 / best_k for key in FSA_100G})
    variants["who_energy_shares"] = who_score(n)

    out_scores, report = {}, {"best_constant_mass_g": best_k,
                              "agreement_vs_mass": {str(k): v for k, v in fit.items() if k % 50 == 0}}
    for v, x in variants.items():
        released = released_who if v.startswith("who") else released_fsa
        report[v] = agreement(x[used], released[used])
        full = np.full(data.n_courses, np.nan)
        full[idx] = x
        out_scores[v] = full
    return out_scores, report


def meal_mean(course_values: np.ndarray, data) -> np.ndarray:
    mc = data.meal_course
    return np.asarray(mc @ np.nan_to_num(course_values)).ravel() / np.asarray(mc.sum(axis=1)).ravel()


def expo_gap(top: np.ndarray, healthier: np.ndarray) -> float:
    utility = 1.0 / np.arange(1, top.shape[1] + 1)
    in_a = healthier[top]
    a, b = (utility * in_a).sum(), (utility * ~in_a).sum()
    return float((a - b) / (a + b))


def expo_gap_variants(top: np.ndarray, fsa: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(fsa)
    stable = np.zeros(n, bool)
    stable[np.argsort(fsa, kind="stable")[: n // 2]] = True
    draws = []
    for _ in range(N_TIE_DRAWS):
        order = np.lexsort((rng.permutation(n), fsa))
        h = np.zeros(n, bool)
        h[order[: n // 2]] = True
        draws.append(expo_gap(top, h))
    cut = np.sort(fsa)[n // 2 - 1]
    strict_a, strict_b = fsa < cut, fsa > cut
    utility = 1.0 / np.arange(1, top.shape[1] + 1)
    a, b = (utility * strict_a[top]).sum(), (utility * strict_b[top]).sum()
    return {"stable": expo_gap(top, stable), "random_ties_min": float(min(draws)),
            "random_ties_max": float(max(draws)), "random_ties_mean": float(np.mean(draws)),
            "strict_no_ties": float((a - b) / (a + b))}


def health_of_top(top, users, meal_score, history_all, history_train) -> dict:
    m = meal_score[top].mean(axis=1)
    return {"mean": float(m.mean()), "delta_vs_released_user_fsa": float((m - history_all[users]).mean()),
            "delta_vs_train_history": float(np.nanmean(m - history_train[users]))}


def main() -> None:
    runs = summaries()
    rng = np.random.default_rng(0)
    result = {}
    for name in DATASETS:
        base = get("orig", name)
        recomputed, report = course_scores(base, name)
        meal_scores = {"fsa_released": base.meal_fsa, "who_released": base.meal_who,
                       **{v: meal_mean(x, base) for v, x in recomputed.items()}}
        print(f"\n== {name}: reproducing course scores (courses used in meals)")
        for v in recomputed:
            print(f"  {v:<28} {report[v]}")

        # (a) less-healthy phenomenon: interaction-weighted mean vs catalogue mean.
        pos = sum(base.user_meal[s] for s in ("train", "tune", "test"))
        weights = np.asarray(pos.sum(axis=0)).ravel()
        phenomenon = {}
        for v, s in meal_scores.items():
            inter, cat = float(np.average(s, weights=weights)), float(s.mean())
            lower_is_healthier = v.startswith("fsa")
            phenomenon[v] = {"interacted": inter, "catalogue": cat, "difference_in_sd": (inter - cat) / float(s.std()),
                             "interacted_less_healthy": bool(inter > cat if lower_is_healthier else inter < cat)}
            print(f"  phenomenon {v:<28} interacted {inter:.3f} vs catalogue {cat:.3f} "
                  f"({phenomenon[v]['difference_in_sd']:+.3f} sd) -> less healthy: {phenomenon[v]['interacted_less_healthy']}")

        # (b, c) models and baselines on every protocol.
        models = {}
        for protocol in PROTOCOLS:
            data = get(protocol, name)
            train, test = data.user_meal["train"], data.user_meal["test"]
            users = np.flatnonzero(test.getnnz(axis=1))
            n_train = np.asarray(train.sum(axis=1)).ravel()
            hist_train = np.asarray(train @ base.meal_fsa).ravel() / np.where(n_train > 0, n_train, np.nan)
            sources = {f"baseline/{b}": [fn(data)] for b, fn in BASELINES.items()}
            for model in MODELS:
                seeds = sorted(runs.query("protocol == @protocol and dataset == @name and model == @model")["seed"])
                sources[model] = [scores(protocol, name, model, s) for s in seeds]
            for src, score_list in sources.items():
                rows = []
                for s in score_list:
                    top = topk(mask_scores(s[users], train[users]), K)
                    rows.append({**expo_gap_variants(top, base.meal_fsa, rng),
                                 **{f"fsa_{k}": v for k, v in health_of_top(
                                     top, users, base.meal_fsa, base.user_fsa, hist_train).items()},
                                 "expo_gap_who_released": expo_gap(top, base.meal_who >= np.sort(base.meal_who)[len(base.meal_who) // 2]),
                                 "who_recomputed_mean": float(meal_scores["who_energy_shares"][top].mean())})
                agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                agg["random_ties_min"] = float(min(r["random_ties_min"] for r in rows))
                agg["random_ties_max"] = float(max(r["random_ties_max"] for r in rows))
                agg["n"] = len(rows)
                agg["sign_flips_with_ties"] = bool(agg["random_ties_min"] < 0 < agg["random_ties_max"])
                models.setdefault(protocol, {})[src] = agg
                print(f"  {protocol:<4} {src:<26} ExpoGap stable {agg['stable']:+.3f} ties [{agg['random_ties_min']:+.3f}, "
                      f"{agg['random_ties_max']:+.3f}] strict {agg['strict_no_ties']:+.3f} | dFSA released "
                      f"{agg['fsa_delta_vs_released_user_fsa']:+.3f} own-train {agg['fsa_delta_vs_train_history']:+.3f}", flush=True)
        result[name] = {"reproduction": report, "less_healthy_phenomenon": phenomenon,
                        "tied_at_median": int((base.meal_fsa == np.sort(base.meal_fsa)[len(base.meal_fsa) // 2 - 1]).sum()),
                        "models": models}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
