"""The ten highest-degree nodes of each real network: Degree's own top 10 (its seeds as A under the
A-first block sequence at k = 10, run 0), the edges among them (of 45 possible), the sum of their
degrees and the number of distinct nodes next to at least one of them.

Writes out/top_degree_group.txt.
Usage: python3 mechanisms/top_degree_group.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("mechanisms")


def group_line(network: str) -> str:
    graph = s.load_graph(network)
    top = np.asarray(s.own_top_k(s.deterministic_matches(network, 10))["degree"], dtype=np.int64)
    neighbours = s.neighbours_of(graph, top)
    edges = int(np.isin(neighbours, top).sum()) // 2
    return (f"{network:14s} edges among the top 10: {edges:2d}/45  degree sum {int(graph.degree[top].sum()):6d}  "
            f"distinct neighbours {np.unique(neighbours).size:6d}")


def main() -> None:
    text = "\n".join(group_line(network) for network in s.REAL) + "\n"
    (OUT / "top_degree_group.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
