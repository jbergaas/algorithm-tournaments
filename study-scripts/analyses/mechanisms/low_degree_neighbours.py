"""Low-degree neighbours of each structural algorithm's own top k.

For the own top k of an algorithm (its seeds as A under the A-first block sequence, run 0), the
reach is the set of non-seed nodes next to a seed; inv_reach sums 1/degree over the reach, so a
neighbour counts more the fewer neighbours it has. Compares PageRank with Degree per budget.

Writes out/low_degree_neighbours.csv (graph_name, k, alg, reach, inv_reach) and
out/low_degree_neighbours.txt.
Usage: python3 mechanisms/low_degree_neighbours.py [workers]
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("mechanisms")


def network_rows(network: str) -> list[dict]:
    graph = s.load_graph(network)
    degree = graph.degree
    rows = []
    for k in s.BUDGETS:
        top = s.own_top_k(s.deterministic_matches(network, k))
        for alg in s.STRUCTURAL:
            seeds = np.asarray(top[alg], dtype=np.int64)
            reach = np.zeros(graph.n, bool)
            reach[s.neighbours_of(graph, seeds)] = True
            reach[seeds] = False
            rows.append(dict(graph_name=network, k=k, alg=alg, reach=int(reach.sum()),
                             inv_reach=float((1 / degree[reach]).sum())))
    return rows


def comparison(table: pd.DataFrame) -> list[str]:
    wide = table.pivot_table(index=["graph_name", "k"], columns="alg", values="inv_reach").reset_index()
    wide["diff"] = wide.pagerank - wide.degree
    lines = ["PageRank vs Degree, sum of 1/degree over the non-seed neighbours of the own top k:"]
    for label, nets in (("all 56 networks", s.NETWORKS), ("11 real networks", s.REAL)):
        for k, x in wide[wide.graph_name.isin(nets)].groupby("k"):
            lines.append(f"  {label}, k={k:2d}: PageRank higher on {int((x['diff'] > 1e-9).sum())}, "
                         f"equal on {int((x['diff'].abs() <= 1e-9).sum())}, lower on {int((x['diff'] < -1e-9).sum())}")
    return lines


def main(workers: int) -> None:
    order = sorted(s.NETWORKS, key=lambda g: g != "com-Youtube")
    with Pool(workers) as pool:
        parts = dict(zip(order, pool.map(network_rows, order, chunksize=1)))
    table = pd.DataFrame([row for network in s.NETWORKS for row in parts[network]])
    table.to_csv(OUT / "low_degree_neighbours.csv", index=False)
    text = "\n".join(comparison(table)) + "\n"
    (OUT / "low_degree_neighbours.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
