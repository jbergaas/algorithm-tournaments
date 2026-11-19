"""Distance race of every replayed match: race_b = (nodes strictly nearer, in hops, to B's seeds
minus nodes strictly nearer to A's seeds) / n. It depends on the seeds only, not on the threshold
or the variant.

Writes out/race_match.csv (network, k, match index of variants.MATCHES, race_b).
Usage: python3 diffusion_sensitivity/race.py [workers] [networks...]
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402  (before numba: it sets numba's cache folder)

import numba as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import variants as v  # noqa: E402

OUT = s.out_dir("diffusion_sensitivity")


@nb.njit(cache=True)
def hop_distance(indptr, indices, seeds):
    """Breadth-first distance of every node from the nearest seed."""
    n = indptr.shape[0] - 1
    dist = np.full(n, np.iinfo(np.int32).max, np.int32)
    queue = np.empty(n, np.int64)
    head = 0
    tail = 0
    for x in seeds:
        if dist[x] != 0:
            dist[x] = 0
            queue[tail] = x
            tail += 1
    while head < tail:
        u = queue[head]
        head += 1
        du = dist[u] + 1
        for p in range(indptr[u], indptr[u + 1]):
            w = indices[p]
            if dist[w] > du:
                dist[w] = du
                queue[tail] = w
                tail += 1
    return dist


def one_network(network: str) -> list[dict]:
    graph = s.load_graph(network)
    rows = []
    for k in v.BUDGETS:
        m = s.deterministic_matches(network, k, self_play=True)
        seeds = {(r.algorithm_a, r.algorithm_b, r.draft_order): (r.seeds_a, r.seeds_b) for r in m.itertuples()}
        cache: dict = {}

        def distance(nodes: list[int]) -> np.ndarray:
            key = tuple(sorted(int(x) for x in nodes))
            if key not in cache:
                cache[key] = hop_distance(graph.indptr, graph.indices, np.asarray(key, np.int64))
            return cache[key]

        for mi, match in enumerate(v.MATCHES):
            da, db = distance(seeds[match][0]), distance(seeds[match][1])
            rows.append(dict(network=network, k=k, mi=mi,
                             race_b=(int((db < da).sum()) - int((da < db).sum())) / graph.n))
    return rows


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    networks = sys.argv[2:] or v.NETWORKS
    with Pool(workers) as pool:
        result = pool.map(one_network, networks, chunksize=1)
    pd.DataFrame([r for rows in result for r in rows]).to_csv(OUT / "race_match.csv", index=False)


if __name__ == "__main__":
    main()
