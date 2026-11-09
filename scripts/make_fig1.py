"""Figure 1: best published Recall@20 vs. the training-free lookup probe vs. the best model under P1.

Small multiples, one panel per dataset with its own y-axis (the five datasets differ by an
order of magnitude). Our numbers come from results/; published numbers are the highest we found
on the same split (see the paper).

    python scripts/make_fig1.py   # writes figs/fig1_lookup.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs"

# Highest published test Recall@20 on the released split.
PUBLISHED = {
    "MealRec+H": (0.3753, "CrossCBR"),   # Li et al., SIGIR 2024, Table 6
    "MealRec+L": (0.1920, "CrossCBR"),
    "Youshu": (0.3245, "DGMAE"),         # as reported in BunCa (arXiv 2408.08906), Table 2
    "NetEase": (0.0947, "TriadCBR"),     # Shen & Wu, Electronics 2026, Table 2
    "iFashion": (0.1562, "TriadCBR"),
}
# Palette slots 1-3 of the dataviz reference palette (validated all-pairs, light mode).
COLORS = {"published": "#2a78d6", "canary": "#eb6834", "clean": "#1baf7a"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def best_clean(row: dict, tuned: list[dict], name: str) -> tuple[float, str]:
    """Best test R@20 under P1 among tuned GNNs, BundleComp and (meal) CompMeal."""
    cands = [(t["gnn_r20"], t["gnn"]) for t in tuned if t["dataset"] == name]
    cands.append((row["bundlecomp_p1"], "BundleComp"))
    if "compmeal_p1" in row:
        cands.append((row["compmeal_p1"], "CompMeal"))
    return max(cands)


def main() -> None:
    s = json.loads((ROOT / "results" / "g5_summary.json").read_text())
    names = list(PUBLISHED)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.edgecolor": MUTED,
                         "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, len(names), figsize=(7.0, 1.9), constrained_layout=True)
    for ax, name in zip(axes, names):
        row = s["table"][name]
        clean, clean_by = best_clean(row, s["tests_vs_tuned"], name)
        bars = [("published", PUBLISHED[name][0], PUBLISHED[name][1]),
                ("canary", row["canary_orig"], ""),
                ("clean", clean, clean_by)]
        top = max(v for _, v, _ in bars)
        for x, (key, v, _) in enumerate(bars):
            ax.bar(x, v, width=0.72, color=COLORS[key], edgecolor="white", linewidth=1.5,
                   hatch="////" if key == "canary" else None, zorder=2)
            ax.text(x, v + top * 0.03, f"{v:.3f}", ha="center", va="bottom", color=TEXT, fontsize=6.5)
            if bars[x][2]:  # model name above the value; the canary needs none (legend)
                ax.text(x, v + top * 0.13, bars[x][2], ha="center", va="bottom", color=MUTED, fontsize=5.2)
        ax.set_title(name, color=TEXT, fontsize=8, pad=4)
        ax.set_ylim(0, top * 1.32)
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.tick_params(axis="y", labelsize=6, length=0)
    axes[0].set_ylabel("Test Recall@20")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS["published"]),
               plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["canary"], hatch="////", edgecolor="white"),
               plt.Rectangle((0, 0), 1, 1, color=COLORS["clean"])]
    fig.legend(handles, ["Best published model (released split)", "Lookup probe (released split, no training)",
                         "Best model under the clean protocol P1"],
               loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.09))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_lookup.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}/fig1_lookup.{{pdf,png}}")


if __name__ == "__main__":
    main()
