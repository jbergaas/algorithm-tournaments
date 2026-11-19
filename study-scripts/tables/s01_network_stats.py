"""Network statistics used as context columns of the result tables.

Reads the graphs in data/ and average_path_lengths.csv; writes data/network_stats.csv with, per
network: nodes, edges, components, average degree (2m/n), maximum degree, coefficient of variation
of the degrees, share of nodes in the largest component, share of degree-1 nodes (%), average
shortest-path length, the share of nodes in or next to the 2k highest-degree nodes for each
budget k (%), and the largest BFS distance from the 40 highest-degree nodes.

The average shortest-path length is not recomputed here. study-scripts/README.md says how
average_path_lengths.csv was made.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from common import AVG_PATH_CSV, BUDGETS, NETWORKS, OUT_DATA, output_path, read_graph


def bfs_distances(n: int, start: np.ndarray, neighbours: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Distance of every node from the nearest source; -1 if unreachable."""
    distance = np.full(n, -1, dtype=np.int64)
    distance[sources] = 0
    frontier = np.asarray(sources)
    depth = 0
    while len(frontier):
        reached = neighbours[np.concatenate([np.arange(start[u], start[u + 1]) for u in frontier])]
        reached = np.unique(reached)
        reached = reached[distance[reached] < 0]
        depth += 1
        distance[reached] = depth
        frontier = reached
    return distance


def reach_of_top_nodes(n: int, top: np.ndarray, edge_u: np.ndarray, edge_v: np.ndarray) -> float:
    """Percentage of nodes that are in `top` or adjacent to a node in it."""
    in_top = np.zeros(n, bool)
    in_top[top] = True
    near = np.zeros(n, bool)
    near[edge_v[in_top[edge_u]]] = True
    near[edge_u[in_top[edge_v]]] = True
    near |= in_top
    return 100 * near.mean()


def network_row(name: str, avg_path: float) -> dict:
    n, start, neighbours, edge_u, edge_v = read_graph(name)
    m = len(edge_u)
    degree = np.diff(start)
    adjacency = csr_matrix((np.ones(len(neighbours), dtype=np.int8), neighbours, start), shape=(n, n))
    components, labels = connected_components(adjacency, directed=False)
    by_degree = np.argsort(-degree, kind="stable")
    row = dict(network=name, nodes=n, edges=m, components=components, avg_deg=2 * m / n,
               max_deg=int(degree.max()), deg_cv=degree.std() / degree.mean(),
               largest=np.bincount(labels).max() / n, deg1_pct=100 * np.mean(degree == 1),
               avg_path=avg_path)
    for k in BUDGETS:
        row[f"reach_top2k_k{k}_pct"] = reach_of_top_nodes(n, by_degree[:2 * k], edge_u, edge_v)
    row["dist_from_top40"] = int(bfs_distances(n, start, neighbours, by_degree[:40]).max())
    return row


def main() -> None:
    avg_paths = pd.read_csv(AVG_PATH_CSV).drop_duplicates("graph").set_index("graph").avg_path
    rows = []
    for name in NETWORKS:
        rows.append(network_row(name, float(avg_paths[name])))
        print(name, "done", flush=True)
    stats = pd.DataFrame(rows)
    stats.to_csv(output_path(OUT_DATA, "network_stats.csv"), index=False)
    pd.set_option("display.width", 250)
    print(stats.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
