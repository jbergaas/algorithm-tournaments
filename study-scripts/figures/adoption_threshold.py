"""Heatmap of the median additional acquisition at k=20 under snake, one row per network and one column
per threshold, with the collapse marked.

Each cell is the median of the 20 ordered cross-algorithm condition means of q, the share of all
nodes acquired beyond the 2k seeds. The values come from the cross-play condition summaries of the
run root set in tables/common.py (STUDY_RUN_ROOT). The collapse is the first R3 threshold above
the last R2 threshold at k=20 (regime_first_R3 in data/collapse.csv, written by
tables/s02_medians.py). A black line marks it before that threshold, and rows are sorted by it.

Usage: python3 adoption_threshold.py <tables output folder> <out.pdf>
Reads <tables output folder>/data/collapse.csv. Writes <out>.pdf, a PNG preview, the plotted values
as <out>.csv and the collapse threshold per network as <out>_collapse.csv.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from netstyle import NETWORKS, THETA_FRACTION, THETAS, cross_play_summaries, with_additional_adoption

INK = "#1f2328"
MUTED = "#57606a"


def snake_medians() -> pd.DataFrame:
    """Median of 100 q (in % of the nodes) over the 20 ordered cross-algorithm conditions at k=20 under
    snake, per network and threshold.

    data/medians.csv of s02 holds 100 times the median of q instead. The two can differ in the last
    bits.
    """
    summaries = with_additional_adoption(cross_play_summaries())
    snake20 = summaries[~summaries.is_self_play & (summaries.k == 20) & (summaries.allocation_order == "snake")]
    medians = snake20.assign(q_pct=100 * snake20.q).groupby(["graph_name", "theta"]).agg(
        median_q_pct=("q_pct", "median"), conditions=("q_pct", "size")).reset_index()
    assert (medians.conditions == 20).all() and len(medians) == len(NETWORKS) * len(THETAS)
    return medians


def draw(values, row_names: list, collapse: pd.Series, out: Path) -> None:
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            v = values[r, c]
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 55 else INK)
        if pd.notna(collapse[row_names[r]]):
            x = THETAS.index(collapse[row_names[r]]) - 0.5
            ax.plot([x, x], [r - 0.5, r + 0.5], color=INK, lw=2.0, solid_capstyle="butt")
    ax.set_xticks(range(len(THETAS)))
    ax.set_xticklabels([THETA_FRACTION[t] for t in THETAS], fontsize=8, color=MUTED, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=8.5, color=INK)
    ax.set_xlabel(r"Threshold $\theta$", fontsize=9, color=INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)
    colorbar.set_label("Additional acquisition (% of nodes)", fontsize=8.5, color=INK)
    colorbar.ax.tick_params(labelsize=8, length=2, colors=MUTED)
    colorbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=180)


def main(tables_out: str, output: str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    medians = snake_medians()
    collapse = pd.read_csv(Path(tables_out) / "data" / "collapse.csv", index_col=0).regime_first_R3
    # Networks that never collapse (empty regime_first_R3) go last; ties are broken by name.
    row_names = sorted(NETWORKS, key=lambda n: (collapse[n] if pd.notna(collapse[n]) else 9, n.lower()))
    heat = medians.pivot(index="graph_name", columns="theta", values="median_q_pct").loc[row_names, THETAS]
    medians[["graph_name", "theta", "median_q_pct"]].to_csv(out.with_suffix(".csv"), index=False)
    pd.DataFrame({"graph_name": row_names, "collapse_upper_theta": [collapse[n] for n in row_names]}).to_csv(
        out.with_name(out.stem + "_collapse.csv"), index=False)
    draw(heat.to_numpy(), row_names, collapse, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
