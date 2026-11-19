"""The test based on Watts's condition against the observed collapse (Results, nodes that need one
owned neighbour, and its table).

1. For each cutoff c (1, 5, 10, 25 and 50% of the nodes): the last sampled threshold at which the
   largest cluster of one-neighbour nodes still holds at least c of the network, against the last
   threshold before the collapse of every network and budget. Counted: the cases where the spread
   lasts at least up to that threshold, and those where it lasts more than one step of the
   threshold grid beyond it.
2. The largest cluster on the real networks at theta = 1/8.
3. Real networks at k = 20, at the last threshold before the collapse: the share of one-neighbour
   nodes against the median adoption without the seeds. Where the adoption is larger, at least the
   difference are nodes that needed two or more owned neighbours, in at least half the conditions.
4. The table rows at k = 20 and c = 10%: per group of networks, the gap in grid steps (range and
   median), and at the last threshold before the collapse the share of one-neighbour nodes and the
   largest cluster. er-d8 and the uneven graphs include the networks of the size check.

Reads out/one_neighbour.csv and collapse/out/observed_collapse.csv. Writes out/summary.txt (also
printed).
Usage: python3 watts/summary.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("watts")
CUTOFFS = (0.01, 0.05, 0.10, 0.25, 0.50)
SEEDS_K20 = 40
GROUPS = [("er-d4", "er-d4"), ("er-d8", "er-d8"), ("er-d16", "er-d16"), ("er-d32", "er-d32"),
          ("ws-p1.0", "ws-p1.0"), ("Uneven (cl)", "cl-"), ("ws-p0.1", "ws-p0.1"), ("ws-p0.01", "ws-p0.01"),
          ("Coauthorship (collab)", "collab-"), ("Real", "real")]


def step(theta: float) -> int:
    return int(np.argmin([abs(theta - t) for t in s.THETAS]))


def family(network: str) -> str:
    """real, or the generated setting with the size-check suffix removed."""
    if network in s.REAL:
        return "real"
    return re.sub(r"(-n\d+)?-i\d+$", "", network)


def gaps(cluster: pd.DataFrame, collapse: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Steps from the last threshold whose largest cluster holds at least the cutoff to the last
    threshold before the collapse, per network and budget with a collapse."""
    last = cluster[cluster.largest_cluster_share >= cutoff].groupby("graph").theta.max()
    c = collapse.dropna(subset=["collapse_lo"])
    return pd.DataFrame({"graph": c.graph, "k": c.k, "before": c.collapse_lo, "last": c.graph.map(last),
                         "steps": [step(b) - step(x) for b, x in zip(c.collapse_lo, c.graph.map(last))]})


def at(cluster: pd.DataFrame, network: str, theta: float) -> pd.Series:
    return cluster[(cluster.graph == network) & np.isclose(cluster.theta, theta)].iloc[0]


def cutoff_lines(cluster: pd.DataFrame, collapse: pd.DataFrame) -> list[str]:
    lines = ["1. Cutoff check, all networks and budgets",
             "   cutoff  cases  collapse at/after it  more than one step after"]
    for cutoff in CUTOFFS:
        d = gaps(cluster, collapse, cutoff)
        lines.append(f"   {cutoff:6.0%}  {len(d):5d}  {int((d.steps >= 0).sum()):20d}  {int((d.steps > 1).sum()):24d}")
    real = cluster[cluster.graph.isin(s.REAL) & np.isclose(cluster.theta, 1 / 8)]
    largest = 100 * real.largest_cluster_share.max()
    lines += ["", f"2. Real networks at theta = 1/8: largest cluster at most {largest:.2f}%"]
    return lines


def real_lines(cluster: pd.DataFrame, collapse: pd.DataFrame) -> list[str]:
    lines = ["", "3. Real networks, k = 20, last threshold before the collapse",
             "   network        theta  one-neighbour  largest cluster  median adoption  seeds"
             "  needed two or more (at least)"]
    c = collapse[collapse.k == 20].set_index("graph")
    for network in sorted(s.REAL):
        theta, adoption = c.loc[network, "collapse_lo"], c.loc[network, "median_lo"]
        x = at(cluster, network, theta)
        seeds = SEEDS_K20 / x.n
        lower = adoption - seeds - x.one_neighbour_share
        lines.append(f"   {network:14s} {theta:6.4f}  {100 * x.one_neighbour_share:12.1f}%  "
                     f"{100 * x.largest_cluster_share:14.2f}%  {100 * adoption:14.1f}%  {100 * seeds:4.2f}%  "
                     f"{100 * max(lower, 0):8.1f}%")
    return lines


def table_lines(cluster: pd.DataFrame, collapse: pd.DataFrame) -> list[str]:
    d = gaps(cluster, collapse, 0.10)
    d = d[d.k == 20].copy()
    d["family"] = d.graph.map(family)
    d["one"] = [at(cluster, r.graph, r.before).one_neighbour_share for r in d.itertuples()]
    d["cluster"] = [at(cluster, r.graph, r.before).largest_cluster_share for r in d.itertuples()]
    lines = ["", "4. Table rows: k = 20, cutoff 10%"]
    for name, prefix in GROUPS:
        g = d[d.family.str.startswith(prefix)] if prefix.endswith("-") else d[d.family == prefix]
        lines.append(f"   {name:22s} n={len(g):2d}  steps {g.steps.min()}-{g.steps.max()} (median {g.steps.median():g})"
                     f"  one-neighbour {100 * g.one.min():.1f}-{100 * g.one.max():.1f}%"
                     f"  largest cluster max {100 * g.cluster.max():.1f}%")
    return lines


def main() -> None:
    cluster = pd.read_csv(OUT / "one_neighbour.csv")
    collapse = pd.read_csv(s.out_dir("collapse") / "observed_collapse.csv")
    lines = (cutoff_lines(cluster, collapse) + real_lines(cluster, collapse)
             + table_lines(cluster, collapse))
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
