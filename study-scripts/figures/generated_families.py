"""Small drawn examples of three of the generated families, 30 nodes each, with the family's one
parameter set low (top row) and high (bottom row). Dot area grows with the number of neighbours.
Illustration only: the generated networks of the study have 10,000 or 50,000 nodes and average
degree 8 (random graphs: 4 to 32).

Usage: python3 generated_families.py <output-without-extension>   (writes .pdf and .png)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

INK, MUTED, NODE, EDGE = "#1f2328", "#57606a", "#2f6db5", "#9aa4ae"
N = 30


def random_graph(avg):
    return lambda s: nx.gnm_random_graph(N, int(avg * N / 2), seed=s)


def uneven_graph(gamma, avg=4):
    def make(s):
        w = np.arange(1, N + 1) ** (-1 / (gamma - 1))
        w = np.minimum(w / w.mean() * avg, N - 1)
        return nx.expected_degree_graph(list(w), seed=s, selfloops=False)
    return make


def ring_graph(p):
    return lambda s: nx.watts_strogatz_graph(N, 4, p, seed=s)


def ring_dist(u, v):
    return min((u - v) % N, (v - u) % N)


def pick(make, ok):
    for s in range(500):
        g = make(s)
        if ok(g):
            return g
    raise SystemExit("no suitable example graph")


def maxdeg(g):
    return max(dict(g.degree()).values())


def shortcuts(g):
    return sum(1 for u, v in g.edges() if ring_dist(u, v) > 2)


def draw(ax, g, ring):
    deg = dict(g.degree())
    if ring:
        pos = nx.circular_layout(sorted(g.nodes()))
        near2 = [(u, v) if (v - u) % N == 2 else (v, u) for u, v in g.edges() if ring_dist(u, v) == 2]
        other = [e for e in g.edges() if ring_dist(*e) != 2]
        nx.draw_networkx_edges(g, pos, edgelist=other, ax=ax, edge_color=EDGE, width=0.7)
        nx.draw_networkx_edges(g, pos, edgelist=near2, ax=ax, edge_color=EDGE, width=0.7, arrows=True,
                               arrowstyle="-", connectionstyle="arc3,rad=0.9", node_size=0)
    else:
        pos = nx.kamada_kawai_layout(g)
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color=EDGE, width=0.7)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=NODE, linewidths=0.6, edgecolors="white",
                           node_size=[7 + 2.6 * deg[v] ** 1.6 for v in g.nodes()])
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.08)


def main(out):
    cols = [
        ("Random graph", "parameter: average degree", False,
         (pick(random_graph(3), nx.is_connected), "average degree 3"),
         (pick(random_graph(6), nx.is_connected), "average degree 6")),
        ("Uneven graph", "parameter: unevenness", False,
         (pick(uneven_graph(3.0), lambda g: nx.is_connected(g) and 7 <= maxdeg(g) <= 9), "mildly uneven"),
         (pick(uneven_graph(2.2), lambda g: nx.is_connected(g) and maxdeg(g) >= 14), "strongly uneven")),
        ("Ring graph", "parameter: edges moved", True,
         (pick(ring_graph(0.05), lambda g: 2 <= shortcuts(g) <= 3), "few edges moved"),
         (pick(ring_graph(1.0), lambda g: True), "every edge moved")),
    ]
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.6))
    for c, (title, param, ring, low, high) in enumerate(cols):
        for r, (g, label) in enumerate((low, high)):
            ax = axes[r][c]
            draw(ax, g, ring)
            ax.text(0.5, -0.02, label, transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
                    color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.88), h_pad=1.4, w_pad=0.8)
    for c, (title, param, *_rest) in enumerate(cols):
        box = axes[0][c].get_position()
        x = (box.x0 + box.x1) / 2
        fig.text(x, 0.985, title, ha="center", va="top", fontsize=11, color=INK)
        fig.text(x, 0.935, param, ha="center", va="top", fontsize=9.5, color=MUTED)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=170)


if __name__ == "__main__":
    main(sys.argv[1])
