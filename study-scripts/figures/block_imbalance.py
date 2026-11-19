"""Block imbalance against how often swapping the block order changes the winner, one point per
network.

The block imbalance of a budget k is the degree sum of the k highest-degree nodes minus that of
the next k, divided by the sum of both (degree descending, node ID ascending). A network's value
is the mean over the four budgets, computed from data/<name>.mtx. The y value is the share of the
624 settings per network (12 ordered pairs of different deterministic algorithms, four budgets,
13 positive thresholds) in which B-first block has a different outcome than A-first block. It is
read from the outcome-change table written by tables/s04_sequences.py.

Usage: python3 block_imbalance.py <outcome_changes.csv> <out.pdf>
Writes <out>.pdf, a PNG preview and the plotted values as <out>.csv.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from netstyle import BUDGETS, NETWORKS, STYLE, read_graph

INK, MUTED = "#222222", "#666666"
# Label offsets in points where the default (right, slightly up) would collide.
LABEL_OFFSET = {"Erdos02": (-22, 7), "ego-Facebook": (-14, -14),
                "soc-Epinions1": (-72, 4), "email-Enron": (6, -10)}


def block_imbalance(name: str) -> float:
    n, start, *_ = read_graph(name)
    degree = np.diff(start)
    by_degree = np.lexsort((np.arange(n), -degree))
    values = []
    for k in BUDGETS:
        first, second = degree[by_degree[:k]].sum(), degree[by_degree[k:2 * k]].sum()
        values.append((first - second) / (first + second))
    return float(np.mean(values))


def main(changes_csv: str, output: str) -> None:
    out = Path(output)
    changes = pd.read_csv(changes_csv).set_index("network")
    rows = []
    for name in NETWORKS:
        c = changes.loc[name]
        rows.append(dict(network=name, block_imbalance=block_imbalance(name),
                         changed_pct=100 * c.B_first / c.settings, changed=int(c.B_first), settings=int(c.settings)))
    data = pd.DataFrame(rows)
    data.to_csv(out.with_suffix(".csv"), index=False)

    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for r in data.itertuples():
        colour, marker = STYLE[r.network]
        ax.scatter(r.block_imbalance, r.changed_pct, s=80 if marker == "*" else 46, color=colour, marker=marker,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(r.network, (r.block_imbalance, r.changed_pct), xytext=LABEL_OFFSET.get(r.network, (5, 3)),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("Block imbalance (mean over the four budgets)", fontsize=9, color=INK)
    ax.set_ylabel("Winner changed by swapping\nthe block order (% of settings)", fontsize=9, color=INK)
    ax.set_xlim(0, max(0.5, data.block_imbalance.max() + 0.05))
    ax.set_ylim(0, 105)
    ax.grid(color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=180)
    print(data.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
