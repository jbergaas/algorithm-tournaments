"""Independent check of the engine's PageRank ranking (Appendix B, numerical checks).

PageRank is recomputed with scipy as the engine defines it: damping 0.85, uniform start 1/n,
teleport 0.15/n, stop when the L1 change is at most 1e-9 or after the iteration limit; ties go to
the lower node id. For every network it reports whether the top 40 at 100 iterations equals the
PageRank ranking the engine recorded at k = 20, the iteration at which the tolerance is reached
(limit 500), and whether the top 40 changes when the limit is raised from 100 to 500.

Writes out/pagerank.csv and out/pagerank.txt.
Usage: python3 checks/pagerank.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("checks")
TOLERANCE = 1e-9
TOP = 40


def pagerank(graph: s.Graph, max_iterations: int) -> tuple[np.ndarray, int | None]:
    """Scores after convergence or the limit, and the iteration that met the tolerance (or None)."""
    n = graph.n
    adjacency = csr_matrix((np.ones(len(graph.indices)), graph.indices, graph.indptr), shape=(n, n))
    degree = graph.degree.astype(float)
    rank = np.full(n, 1.0 / n)
    for iteration in range(1, max_iterations + 1):
        following = 0.15 / n + 0.85 * (adjacency.T @ (rank / degree))
        change = np.abs(following - rank).sum()
        rank = following
        if change <= TOLERANCE:
            return rank, iteration
    return rank, None


def top(rank: np.ndarray) -> list[int]:
    return np.lexsort((np.arange(len(rank)), -rank))[:TOP].tolist()


def engine_ranking(network: str) -> list[int]:
    matches = s.deterministic_matches(network, 20)
    return matches[matches.algorithm_a == "pagerank"].ranking_a.iloc[0]


def check(network: str) -> dict:
    graph = s.load_graph(network)
    rank100, _ = pagerank(graph, 100)
    rank500, reached = pagerank(graph, 500)
    return dict(network=network, engine_top40_reproduced=engine_ranking(network) == top(rank100),
                tolerance_reached_at=reached, top40_same_at_500=top(rank100) == top(rank500))


def main() -> None:
    table = pd.DataFrame([check(network) for network in s.REAL + s.GENERATED])
    table.to_csv(OUT / "pagerank.csv", index=False)
    late = table[table.tolerance_reached_at.isna() | (table.tolerance_reached_at > 100)]
    lines = [f"networks: {len(table)}",
             f"engine top 40 reproduced at 100 iterations: {int(table.engine_top40_reproduced.sum())}",
             f"top 40 unchanged at 500 iterations: {int(table.top40_same_at_500.sum())}",
             "tolerance not reached within 100 iterations: "
             + (", ".join(f"{r.network} ({r.tolerance_reached_at:.0f})" for r in late.itertuples()) or "none")]
    (OUT / "pagerank.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
