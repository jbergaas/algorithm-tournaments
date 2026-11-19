"""Nodes that need only one owned neighbour, and the largest connected cluster they form.

A neutral node with d neighbours joins a player once that player owns at least theta * d of them
(and more than the other player), so one owned neighbour is enough exactly when d <= 1 / theta; at
theta = 0 every node is such a node. For every network and sampled threshold this gives the share
of these one-neighbour nodes and the share of the network in the largest connected component of
the subgraph they induce (Watts's vulnerable cluster).

Writes out/one_neighbour.csv: graph, n, theta, one_neighbour_share, largest_cluster_share.
Usage: python3 watts/one_neighbour.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("watts")


def largest_cluster(adjacency: sp.csr_matrix, members: np.ndarray) -> int:
    """Size of the largest connected component of the subgraph induced by the members."""
    sub = adjacency[members][:, members]
    if sub.shape[0] == 0:
        return 0
    _, component = connected_components(sub, directed=False)
    return int(np.bincount(component).max())


def network_rows(network: str) -> list[dict]:
    graph = s.load_graph(network)
    adjacency = sp.csr_matrix((np.ones(graph.indices.size, np.int8), graph.indices, graph.indptr),
                              shape=(graph.n, graph.n))
    degree = graph.degree
    rows = []
    for theta in s.THETAS:
        members = np.ones(graph.n, bool) if theta == 0 else degree <= 1 / theta + 1e-12
        rows.append(dict(graph=network, n=graph.n, theta=theta, one_neighbour_share=members.mean(),
                         largest_cluster_share=largest_cluster(adjacency, members) / graph.n))
    return rows


def main() -> None:
    rows = [row for network in sorted(s.NETWORKS) for row in network_rows(network)]
    pd.DataFrame(rows).to_csv(OUT / "one_neighbour.csv", index=False)


if __name__ == "__main__":
    main()
