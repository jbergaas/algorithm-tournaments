"""Median first-pick advantage per real network under A-first block, strict alternation and snake,
before the collapse (left) and after it (right). The first-pick advantage of a pair of algorithms in
one setting is the first picker's margin averaged over the two role assignments, in percent of the
nodes. The median runs over the deterministic cross-play pairs and the settings on that side of the
collapse.

Usage: python3 first_pick.py <role_adv_by_graph.csv> <out.pdf>
Writes <out>.pdf, a PNG preview and the plotted values as <out>.csv.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from netstyle import NETWORKS

SEQUENCES = [("sequential_a", "A-first block", "#0072B2", "o"),
             ("alternating_a", "strict alternation", "#D55E00", "^"),
             ("snake", "snake", "#009E73", "s")]
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
DODGE = {"sequential_a": 0.2, "alternating_a": 0.0, "snake": -0.2}  # rows are spread so no marker hides another


def main(table: str, out: str) -> None:
    out = Path(out)
    d = pd.read_csv(table)
    d = d[d.graph_name.isin(NETWORKS) & d.allocation_order.isin([s[0] for s in SEQUENCES])]
    wide = d.pivot(index="graph_name", columns="allocation_order", values=["S_med", "N_med"])
    order = wide["S_med"]["sequential_a"].sort_values().index.tolist()
    rows = [dict(network=g, sequence=s, before=wide.loc[g, ("S_med", s)], after=wide.loc[g, ("N_med", s)])
            for g in order for s, *_ in SEQUENCES]
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)

    plt.rcParams.update({"font.size": 9})
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.2, 3.9), sharey=True,
                                      gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.08})
    for ax, column, title in ((left, "S_med", "Before the collapse"), (right, "N_med", "After the collapse")):
        for y, g in enumerate(order):
            values = [wide.loc[g, (column, s)] for s, *_ in SEQUENCES]
            ax.plot([min(values), max(values)], [y, y], color="#cccccc", linewidth=1.2, zorder=1)
        for s, label, colour, marker in SEQUENCES:
            ax.scatter([wide.loc[g, (column, s)] for g in order], [y + DODGE[s] for y in range(len(order))], s=30,
                       color=colour, marker=marker, edgecolor="white", linewidth=0.5, zorder=3, label=label)
        ax.axvline(0, color=MUTED, linewidth=0.8, zorder=0)
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    left.set_yticks(range(len(order)), order, fontsize=8.5, color=INK)
    left.set_xlim(-5, 90)
    right.set_xlim(-3, 7)
    fig.supxlabel("Median first-pick advantage (% of the nodes)", fontsize=9, color=INK, y=0.02)
    left.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=180)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
