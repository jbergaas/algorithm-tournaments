"""Collapse on the generated networks at k = 20 (the table of Results, section on one property at a
time).

Per generated network of the twelve settings in the table: average degree, average clustering,
degree share (the degree sum of the 40 nodes of highest degree as a percentage of the sum of all
degrees, and their mean degree), and the collapse intervals from out/intervals.csv (summary.py):
observed, 2k random seeds alone, and the deterministic matches on the rewired copy. Per setting the
table gives means over the three instances and the intervals, listed per instance when they differ.
The means are taken after rounding per network: average degree and degree share to 2 decimals,
clustering to 3.

Writes out/generated_table.csv (one row per network) and out/generated_table.txt (one row per
setting; also printed).
Usage: python3 collapse/generated_table.py
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("collapse")
SETTINGS = ["er-d4", "er-d8", "er-d16", "er-d32", "cl-g3.0", "cl-g2.5", "cl-g2.2", "ws-p1.0", "ws-p0.1",
            "ws-p0.01", "er-d8-n50000", "cl-g2.5-n50000"]
K = 20


def label(i: int) -> str:
    if i >= len(s.THETAS):
        return ">3/4"
    return f"{s.THETA_LABEL[s.THETAS[i - 1]]}-{s.THETA_LABEL[s.THETAS[i]]}"


def network_row(network: str) -> dict:
    graph = s.load_graph(network)
    degree = graph.degree
    top = np.sort(degree)[::-1][:2 * K]
    g = nx.Graph()
    g.add_nodes_from(range(graph.n))
    rows = np.repeat(np.arange(graph.n), degree)
    upper = graph.indices > rows
    g.add_edges_from(zip(rows[upper].tolist(), graph.indices[upper].tolist()))
    return dict(graph=network, setting=s.setting_of(network), average_degree=round(degree.mean(), 2),
                clustering=round(nx.average_clustering(g), 3), degree_share=round(100 * top.sum() / degree.sum(), 2),
                top40_mean_degree=round(top.mean(), 1))


def intervals_text(values: pd.Series) -> str:
    if values.nunique() == 1:
        return label(values.iloc[0])
    return ", ".join(label(v) for v in values)


def setting_lines(table: pd.DataFrame) -> list[str]:
    lines = ["setting         avg.deg. clust. share(%)  collapse                          random seeds"
             "                      rewired"]
    for setting, x in table.groupby("setting", sort=False):
        lines.append(f"{setting:15s} {x.average_degree.mean():8.2f} {x.clustering.mean():6.3f} "
                     f"{x.degree_share.mean():9.1f}  {intervals_text(x.observed):33s} "
                     f"{intervals_text(x.random_seeds):33s} {intervals_text(x.rewired)}")
    return lines


def main() -> None:
    networks = [f"{setting}-i{i}" for setting in SETTINGS for i in range(3)]
    stats = pd.DataFrame([network_row(n) for n in networks])
    intervals = pd.read_csv(OUT / "intervals.csv")
    intervals = intervals[intervals.k == K][["graph", "observed", "random_seeds", "deterministic", "rewired"]]
    table = stats.merge(intervals, on="graph", how="left", validate="one_to_one")
    table.to_csv(OUT / "generated_table.csv", index=False)
    text = "\n".join(setting_lines(table)) + "\n"
    (OUT / "generated_table.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
