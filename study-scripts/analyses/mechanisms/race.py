"""Distance race of every deterministic match between two different structural algorithms.

Each node counts for the player whose nearest seed is at the smaller hop distance; a node equally
far from both counts for neither. For the seeds a match allocated (run 0, all 56 networks, four
budgets, four sequences), race_b = (nodes nearer to B's seeds - nodes nearer to A's seeds) / n,
so the race leader is B when race_b > 0 and A when it is < 0. The seeds do not depend on the
threshold, so one value serves all thresholds.

Writes out/race_match.csv (graph_name, k, algorithm_a, algorithm_b, allocation_order, race_b).
Usage: python3 mechanisms/race.py [workers]
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("mechanisms")
UNREACHED = np.iinfo(np.int32).max


def distances(graph: s.Graph, seeds: tuple[int, ...]) -> np.ndarray:
    """Hop distance of every node to the nearest seed (breadth-first search)."""
    dist = np.full(graph.n, UNREACHED, dtype=np.int32)
    front = np.unique(np.asarray(seeds, dtype=np.int64))
    dist[front] = 0
    level = 0
    while front.size:
        level += 1
        reached = np.unique(s.neighbours_of(graph, front))
        reached = reached[dist[reached] == UNREACHED]
        dist[reached] = level
        front = reached
    return dist


def cached_distances(graph: s.Graph, seeds: list[int], known: dict) -> np.ndarray:
    key = tuple(sorted(seeds))
    if key not in known:
        known[key] = distances(graph, key)
    return known[key]


def network_rows(network: str) -> list[dict]:
    graph = s.load_graph(network)
    rows = []
    for k in s.BUDGETS:
        known: dict[tuple[int, ...], np.ndarray] = {}
        for m in s.deterministic_matches(network, k).itertuples():
            da, db = cached_distances(graph, m.seeds_a, known), cached_distances(graph, m.seeds_b, known)
            rows.append(dict(graph_name=network, k=k, algorithm_a=m.algorithm_a, algorithm_b=m.algorithm_b,
                             allocation_order=m.draft_order,
                             race_b=(int((db < da).sum()) - int((da < db).sum())) / graph.n))
    return rows


def main(workers: int) -> None:
    order = sorted(s.NETWORKS, key=lambda g: g not in ("com-Youtube", "com-DBLP", "soc-Epinions1"))
    with Pool(workers) as pool:
        parts = dict(zip(order, pool.map(network_rows, order, chunksize=1)))
    rows = [row for network in s.NETWORKS for row in parts[network]]
    pd.DataFrame(rows).to_csv(OUT / "race_match.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
