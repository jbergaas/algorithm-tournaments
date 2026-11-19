"""Two cheap stand-ins for the tournament, played on the network alone at every sampled threshold.

degree_duel     one Degree-vs-Degree match: the 2k nodes of highest degree (ties by lower node id)
                split by strict alternation, ranks 1, 3, 5, ... to A and 2, 4, 6, ... to B
random_seeds    a single player that owns 2k uniformly random nodes and spreads alone; the median
                adoption over three draws (random generator seeded with the CRC-32 of the network
                name, the draws of all budgets made in budget order)

Writes out/graph_only.csv: network, budget, threshold and the adoption (seeds included) of both.
Usage: python3 collapse/degree_duel.py [workers]
"""
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("collapse")
DRAWS = 3


def top_degree(degree: np.ndarray, count: int) -> np.ndarray:
    """The count nodes of highest degree, ties broken by the lower node id."""
    return np.lexsort((np.arange(len(degree)), -degree))[:count]


def one_network(network: str) -> list[dict]:
    graph = s.load_graph(network)
    rng = np.random.default_rng(zlib.crc32(network.encode()))
    rows = []
    for k in s.BUDGETS:
        top = top_degree(graph.degree, 2 * k)
        draws = [rng.choice(graph.n, size=2 * k, replace=False) for _ in range(DRAWS)]
        for theta in s.THETAS:
            rows.append(dict(graph=network, k=k, theta=theta,
                             degree_duel=s.adoption(graph, top[0::2], top[1::2], theta),
                             random_seeds=float(np.median([s.adoption(graph, d, [], theta) for d in draws]))))
    return rows


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    order = sorted(s.NETWORKS, key=lambda x: -s.load_graph(x).indices.size)
    with Pool(workers) as pool:
        parts = pool.map(one_network, order, chunksize=1)
    rows = pd.DataFrame([r for part in parts for r in part])
    rows.to_csv(OUT / "graph_only.csv", index=False)


if __name__ == "__main__":
    main()
