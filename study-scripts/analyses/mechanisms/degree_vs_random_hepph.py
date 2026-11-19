"""Why Degree loses to Random on ca-HepPh at k = 5 and 10.

1. The 20 nodes of highest degree (Degree's ranking: degree descending, lower id first): their
   degrees and how many of the nodes above each one it is linked to (a clique when all).
2. The engine's mean margins against Random on the real networks: every pair of network and
   structural algorithm that loses at least one of its 416 conditions to Random, with its worst
   margin. Degree's mean margin against Random on ca-HepPh per threshold and budget (mean over
   sequences and roles), and its margins at k = 15 and 20 at theta 5/32 and 3/16.
3. Playing alone at theta 5/32 and 3/16: the share of the network that Degree's top k reaches,
   and the same for 100 draws of k random seeds (numpy default_rng(2026), drawn in this order).

Writes out/degree_vs_random_hepph.txt.
Usage: python3 mechanisms/degree_vs_random_hepph.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

NETWORK = "ca-HepPh"
THETAS = (5 / 32, 3 / 16)
DRAWS = 100
OUT = s.out_dir("mechanisms")


def clique_lines(graph: s.Graph, top: np.ndarray) -> list[str]:
    lines = []
    for j, v in enumerate(top):
        neighbours = set(graph.indices[graph.indptr[v]:graph.indptr[v + 1]].tolist())
        links = sum(1 for u in top[:j] if u in neighbours)
        lines.append(f"  rank {j + 1:2d}: node {v}, degree {graph.degree[v]}, linked to {links} of the {j} above")
    return lines


def own_margins(d: pd.DataFrame) -> pd.DataFrame:
    """One row per condition and player: the algorithm, its opponent and its own mean margin."""
    key = ["graph_name", "k", "theta", "allocation_order"]
    a = d[key].assign(alg=d.algorithm_a, opp=d.algorithm_b, margin=-d.mean_margin_b)
    b = d[key].assign(alg=d.algorithm_b, opp=d.algorithm_a, margin=d.mean_margin_b)
    return pd.concat([a, b], ignore_index=True)


def engine_lines() -> list[str]:
    m = own_margins(s.cross_play())
    m = m[(m.opp == "random") & (m.alg != "random") & (m.theta > 0) & m.graph_name.isin(s.REAL)]
    worst = m.groupby(["graph_name", "alg"]).margin.agg(["min", "size", lambda x: int((x < 0).sum())])
    worst.columns = ["worst margin", "conditions", "losses"]
    lines = ["Pairs (network, algorithm) that lose at least one condition to Random:",
             worst[worst.losses > 0].round(4).to_string(), ""]
    degree = m[(m.graph_name == NETWORK) & (m.alg == "degree")]
    lines += [f"Degree vs Random on {NETWORK}: mean margin per threshold (rows) and budget (columns)",
              degree.pivot_table(index="theta", columns="k", values="margin").round(3).to_string(), ""]
    sel = degree[degree.k.isin([15, 20]) & np.isclose(degree.theta.to_numpy()[:, None], THETAS).any(axis=1)]
    lines.append(f"k=15/20 at theta 5/32, 3/16: {len(sel)} conditions, min margin {sel.margin.min():.3f}, "
                 f"mean {sel.margin.mean():.3f}")
    return lines


def alone_lines(graph: s.Graph, order: np.ndarray) -> list[str]:
    rng = np.random.default_rng(2026)
    lines = ["Playing alone: share of the network reached"]
    for theta in THETAS:
        for k in s.BUDGETS:
            owner, rounds, _ = s.propagate(graph, order[:k], [], theta)
            reach = np.array([(s.propagate(graph, rng.choice(graph.n, k, replace=False), [], theta)[0] == 1).mean()
                              for _ in range(DRAWS)])
            lines.append(f"  theta={s.THETA_LABEL[theta]:>5} k={k:2d}: Degree {(owner == 1).mean():.3f} "
                         f"({rounds} rounds); Random mean {reach.mean():.3f}, takes off (>30%) in "
                         f"{(reach > 0.3).mean():.0%} of draws")
    return lines


def main() -> None:
    graph = s.load_graph(NETWORK)
    order = np.argsort(-graph.degree, kind="stable")
    lines = [f"{NETWORK}: {graph.n} nodes, median degree {int(np.median(graph.degree))}", "Top 20 by degree:"]
    lines += clique_lines(graph, order[:20]) + [""]
    lines += engine_lines() + [""]
    lines += alone_lines(graph, order)
    text = "\n".join(lines) + "\n"
    (OUT / "degree_vs_random_hepph.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
