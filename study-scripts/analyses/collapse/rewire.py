"""The deterministic matches replayed on the network and on a copy rewired at random with every
degree kept.

Each network is rewired once by 10 x m attempted double edge swaps (a-b, c-d -> a-d, c-b). A swap
that would make a self-loop or a multi-edge is skipped. The random generator is seeded with the
CRC-32 of the network name. The 48 deterministic matches (12 ordered pairs of different structural
algorithms, four sequences, run 0) of every budget are then played with their stored seeds at every
sampled threshold, on the network and on the rewired copy.

Writes out/deterministic_matches.csv.gz: network, budget, threshold, match and the adoption (seeds
included) on the network ("original") and on the rewired copy ("rewired").
Usage: python3 collapse/rewire.py [workers]
"""
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from numba import types
from numba.typed import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("collapse")


@nb.njit(cache=True)
def _swap(eu, ev, n, attempts, seed):
    """Double edge swaps in place on the edge list (eu[i] < ev[i]); returns the accepted count."""
    np.random.seed(seed)
    m = eu.shape[0]
    present = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i in range(m):
        present[eu[i] * n + ev[i]] = 1
    accepted = 0
    for _ in range(attempts):
        i = np.random.randint(m)
        j = np.random.randint(m)
        if i == j:
            continue
        a, b = eu[i], ev[i]
        c, d = eu[j], ev[j]
        if np.random.random() < 0.5:
            c, d = d, c
        if a == d or c == b:
            continue
        x1, y1 = min(a, d), max(a, d)
        x2, y2 = min(c, b), max(c, b)
        if x1 == x2 and y1 == y2:
            continue
        k1, k2 = x1 * n + y1, x2 * n + y2
        if k1 in present or k2 in present:
            continue
        del present[eu[i] * n + ev[i]]
        del present[eu[j] * n + ev[j]]
        present[k1] = 1
        present[k2] = 1
        eu[i], ev[i] = x1, y1
        eu[j], ev[j] = x2, y2
        accepted += 1
    return accepted


def rewired(network: str, graph: s.Graph) -> s.Graph:
    """Degree-preserving random rewiring of the network."""
    rows = np.repeat(np.arange(graph.n, dtype=np.int64), graph.degree)
    upper = graph.indices > rows
    eu, ev = rows[upper].copy(), graph.indices[upper].copy()
    _swap(eu, ev, graph.n, 10 * len(eu), zlib.crc32(network.encode()) % (2**31))
    src = np.concatenate([eu, ev])
    dst = np.concatenate([ev, eu])
    key = np.unique(src * graph.n + dst)
    src, dst = key // graph.n, key % graph.n
    copy = s.Graph(graph.n, np.searchsorted(src, np.arange(graph.n + 1)).astype(np.int64), dst)
    assert np.array_equal(copy.degree, graph.degree)
    return copy


def one_network(network: str) -> list[dict]:
    graph = s.load_graph(network)
    copy = rewired(network, graph)
    rows = []
    for k in s.BUDGETS:
        for m in s.deterministic_matches(network, k).itertuples():
            for theta in s.THETAS:
                rows.append(dict(graph=network, k=k, theta=theta, algorithm_a=m.algorithm_a,
                                 algorithm_b=m.algorithm_b, draft_order=m.draft_order,
                                 original=s.adoption(graph, m.seeds_a, m.seeds_b, theta),
                                 rewired=s.adoption(copy, m.seeds_a, m.seeds_b, theta)))
    return rows


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    order = sorted(s.NETWORKS, key=lambda x: -s.load_graph(x).indices.size)
    with Pool(workers) as pool:
        parts = pool.map(one_network, order, chunksize=1)
    rows = pd.DataFrame([r for part in parts for r in part])
    rows.to_csv(OUT / "deterministic_matches.csv.gz", index=False)


if __name__ == "__main__":
    main()
