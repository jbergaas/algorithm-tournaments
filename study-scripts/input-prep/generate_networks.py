"""Generate networks in which one property is varied at a time.

Each family keeps the other properties as fixed as the generator allows:
  er  - Erdos-Renyi, mean degree 4, 8, 16, 32: varies the average degree (even degrees, low clustering)
  cl  - Chung-Lu with power-law expected degrees, mean degree 8, exponent 2.2, 2.5, 3.0: varies how
        uneven the degrees are
  ws  - Watts-Strogatz, mean degree 8, rewiring 0.01, 0.1, 1.0: varies the clustering
  collab - coauthorship-like: each "paper" joins its authors in a clique; small papers plus big papers
        of 10, 50, or 200 authors, mean degree 8: varies how much the highest-degree nodes overlap
        (as in ca-HepPh)

Every graph is written like the processed inputs: undirected, no loops or duplicates, isolates
removed, nodes renumbered 0..n-1 in their original order, Matrix Market "coordinate pattern
symmetric" with row > column. The dataset identity follows include/engine/dataset_identity.hpp
(ci-dataset-v1:<file SHA-256>:<population SHA-256>:<n>).

Usage: python3 generate_networks.py <output directory> [--n 10000] [--instances 3]
"""
import argparse
import hashlib
import json
import sys
from array import array
from pathlib import Path

import networkx as nx
import numpy as np


def families(n: int):
    for d in (4, 8, 16, 32):
        yield f"er-d{d}", {"model": "erdos_renyi", "n": n, "mean_degree": d}, \
            lambda seed, d=d: nx.fast_gnp_random_graph(n, d / (n - 1), seed=seed)
    for gamma in (2.2, 2.5, 3.0):
        yield f"cl-g{gamma}", {"model": "chung_lu", "n": n, "mean_degree": 8, "exponent": gamma}, \
            lambda seed, g=gamma: chung_lu(n, 8, g, seed)
    for p in (0.01, 0.1, 1.0):
        yield f"ws-p{p}", {"model": "watts_strogatz", "n": n, "k": 8, "rewiring": p}, \
            lambda seed, p=p: nx.watts_strogatz_graph(n, 8, p, seed=seed)
    for size in (10, 50, 200):
        yield f"collab-s{size}", {"model": "coauthorship", "n": n, "mean_degree": 8, "big_paper_size": size}, \
            lambda seed, size=size: coauthorship(n, 8, size, seed)


def stable_seed(label: str) -> int:
    """Seed derived from the graph label, so that every graph can be regenerated exactly."""
    return int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def coauthorship(n: int, mean_degree: float, big_size: int, seed: int) -> nx.Graph:
    """Projection of "papers" onto their authors, where every paper joins its authors in a clique.
    First, small papers of 2 to 4 authors are added until the mean degree is mean_degree - 2. Every
    author writes at least one, and active authors (Zipf-like weights) write more. Then papers of
    big_size authors, drawn by the same activity weights, are added until the mean degree reaches
    mean_degree. With large big_size, a few big papers turn their authors into high-degree nodes
    that are all connected to each other, as the large collaborations do in ca-HepPh."""
    rng = np.random.default_rng(seed)
    activity = np.arange(1, n + 1, dtype=float) ** -0.6
    activity = rng.permutation(activity / activity.sum())
    g = nx.empty_graph(n)

    def add_paper(authors) -> None:
        g.add_edges_from((int(a), int(b)) for i, a in enumerate(authors) for b in authors[i + 1:])

    for author in rng.permutation(n):
        if g.degree(int(author)) == 0:
            others = rng.choice(n, size=int(rng.integers(1, 4)), replace=False, p=activity)
            add_paper([int(author)] + [int(o) for o in others if o != author])
    while 2 * g.number_of_edges() / n < mean_degree - 2:
        add_paper(rng.choice(n, size=int(rng.integers(2, 5)), replace=False, p=activity))
    while 2 * g.number_of_edges() / n < mean_degree:
        add_paper(rng.choice(n, size=big_size, replace=False, p=activity))
    return g


def chung_lu(n: int, mean_degree: float, gamma: float, seed: int) -> nx.Graph:
    """Expected-degree graph with power-law weights w_i ~ i^(-1/(gamma-1)), scaled to the mean degree,
    capped at sqrt(n * mean_degree) and scaled again, so a few edge probabilities can exceed one;
    networkx treats those as one."""
    ranks = np.arange(1, n + 1, dtype=float)
    weights = ranks ** (-1.0 / (gamma - 1.0))
    weights *= mean_degree / weights.mean()
    weights = np.minimum(weights, np.sqrt(n * mean_degree))
    weights *= mean_degree / weights.mean()
    rng = np.random.default_rng(seed)
    weights = rng.permutation(weights)
    return nx.expected_degree_graph(weights.tolist(), seed=seed, selfloops=False)


def prepare(graph: nx.Graph) -> nx.Graph:
    """Simple undirected graph without isolates, nodes renumbered 0..n-1 in original order."""
    g = nx.Graph(graph)
    g.remove_edges_from(nx.selfloop_edges(g))
    g.remove_nodes_from([v for v, d in dict(g.degree()).items() if d == 0])
    return nx.convert_node_labels_to_integers(g, ordering="sorted")


def write_mtx(g: nx.Graph, path: Path, comment: str) -> None:
    n, m = g.number_of_nodes(), g.number_of_edges()
    edges = sorted((max(u, v) + 1, min(u, v) + 1) for u, v in g.edges())
    with path.open("w") as f:
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write(f"% {comment}\n")
        f.write(f"{n} {n} {m}\n")
        for r, c in edges:
            f.write(f"{r} {c}\n")


def population_sha256(n: int) -> str:
    result = hashlib.sha256()
    for first in range(0, n, 65536):
        values = array("I", range(first, min(n, first + 65536)))
        if sys.byteorder != "little":
            values.byteswap()
        result.update(values.tobytes())
    return result.hexdigest()


def statistics(g: nx.Graph) -> dict:
    deg = np.array([d for _, d in g.degree()], dtype=float)
    n = g.number_of_nodes()
    top = [v for v, _ in sorted(g.degree(), key=lambda x: (-x[1], x[0]))[:20]]
    neighbours = set()
    for v in top:
        neighbours.update(g.neighbors(v))
    neighbours -= set(top)
    return {
        "nodes": n, "edges": g.number_of_edges(), "average_degree": round(deg.mean(), 2),
        "degree_cv": round(deg.std() / deg.mean(), 2), "max_degree": int(deg.max()),
        "clustering": round(nx.average_clustering(g), 3),
        "largest_component": round(max(len(c) for c in nx.connected_components(g)) / n, 3),
        "top20_overlap": round(1 - len(neighbours) / deg[top].sum(), 2),
        "degree_one_share": round(float((deg == 1).mean()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--only", nargs="*", default=None,
                        help="setting names to generate, e.g. er-d8 cl-g2.5 (default: all)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, params, make in families(args.n):
        if args.only and name not in args.only:
            continue
        for instance in range(args.instances):
            seed = stable_seed(f"{name}-i{instance}" if args.n == 10000 else f"{name}-n{args.n}-i{instance}")
            g = prepare(make(seed))
            label = f"{name}-i{instance}" if args.n == 10000 else f"{name}-n{args.n}-i{instance}"
            folder = args.out / label
            folder.mkdir(exist_ok=True)
            path = folder / f"{label}.mtx"
            write_mtx(g, path, f"Generated: {json.dumps(params)}, seed {seed}; isolates removed.")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            n = g.number_of_nodes()
            identity = f"ci-dataset-v1:{sha}:{population_sha256(n)}:{n}"
            stats = statistics(g)
            record = {"name": label, "generator": params, "seed": seed, "dataset_identity": identity,
                      "statistics": stats}
            (folder / "generator.json").write_text(json.dumps(record, indent=2) + "\n")
            summary.append(record)
            print(f"{label:16s} " + " ".join(f"{k}={v}" for k, v in stats.items()))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
