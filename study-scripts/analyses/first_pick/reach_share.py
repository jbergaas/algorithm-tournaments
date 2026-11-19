"""Does the first picker's head start grow during the spread?

The reach of a seed set is the number of nodes next to it that are no player's seed. For every
match between two different structural algorithms (all networks, budgets, sequences and positive
thresholds) this sets the first picker's share of the two players' reach against its share of the
adopted nodes, and averages the latter in bins of the reach share, before and after the collapse.
Bins with fewer than 30 matches are left out.

Writes out/reach.csv (reach of both players per network, budget, sequence and pair) and
out/reach_share.txt (mean share of the adopted nodes per bin, by family before the collapse and
pooled over all networks after it).
Usage: python3 first_pick/reach_share.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402
from common import deterministic, family_of, spreading  # noqa: E402

OUT = s.out_dir("first_pick")
BINS = np.array([0.30, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.70, 0.80])
KEY = ["graph_name", "k", "allocation_order", "algorithm_a", "algorithm_b"]


def reach(graph: s.Graph, seeds: list[int], is_seed: np.ndarray) -> int:
    nodes = s.neighbours_of(graph, seeds)
    return int(np.unique(nodes[~is_seed[nodes]]).size)


def reach_table() -> pd.DataFrame:
    rows = []
    for network in s.NETWORKS:
        graph = s.load_graph(network)
        for k in s.BUDGETS:
            for m in s.deterministic_matches(network, k).itertuples():
                is_seed = np.zeros(graph.n, bool)
                is_seed[m.seeds_a + m.seeds_b] = True
                rows.append(dict(graph_name=network, k=k, allocation_order=m.draft_order, algorithm_a=m.algorithm_a,
                                 algorithm_b=m.algorithm_b, reach_a=reach(graph, m.seeds_a, is_seed),
                                 reach_b=reach(graph, m.seeds_b, is_seed)))
    return pd.DataFrame(rows)


def binned(x: pd.DataFrame) -> pd.Series:
    g = x.groupby("bin").adopted_share.agg(["mean", "size"])
    return 100 * g["mean"][g["size"] >= 30]


def main() -> None:
    table = reach_table()
    table.to_csv(OUT / "reach.csv", index=False)
    d = s.cross_play()
    x = deterministic(d).merge(table, on=KEY).merge(spreading(d), on=["graph_name", "k", "theta"])
    first_is_a = x.allocation_order != "sequential_b"
    reach_a = x.reach_a / (x.reach_a + x.reach_b)
    adopted_a = x.mean_a_fraction / (x.mean_a_fraction + x.mean_b_fraction)
    x["reach_share"] = np.where(first_is_a, reach_a, 1 - reach_a)
    x["adopted_share"] = np.where(first_is_a, adopted_a, 1 - adopted_a)
    x["bin"] = pd.cut(x.reach_share, BINS)
    x["family"] = x.graph_name.map(family_of)
    curves = {f"before, {f}": binned(y) for f, y in x[x.spread].groupby("family")}
    curves["before, all"] = binned(x[x.spread])
    curves["after, all"] = binned(x[~x.spread])
    text = ("First picker's share of the adopted nodes (%) by its share of the seed reach\n"
            + pd.DataFrame(curves).T.round(1).to_string())
    (OUT / "reach_share.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
