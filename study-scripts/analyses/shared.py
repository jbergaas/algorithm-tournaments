"""Paths, study constants and loaders shared by the analysis scripts.

Paths can be set through environment variables:
  STUDY_RUN_ROOT      run root of the study (default <repo>/results/full-study)
  STUDY_DATA          folder with the network files (default <repo>/data)
  STUDY_ANALYSES_OUT  folder that receives <analysis>/out/ and out/cache/ (default: this folder)
  STUDY_TABLES_OUT    output folder of ../tables (default ../tables/out); some analyses read its data/
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("STUDY_REPO", HERE.parents[1]))
RUN_ROOT = Path(os.environ.get("STUDY_RUN_ROOT", REPO / "results" / "full-study"))
DATA = Path(os.environ.get("STUDY_DATA", REPO / "data"))
OUT_ROOT = Path(os.environ.get("STUDY_ANALYSES_OUT", HERE))
TABLES_DATA = Path(os.environ.get("STUDY_TABLES_OUT", HERE.parent / "tables" / "out")) / "data"
CACHE = OUT_ROOT / "out" / "cache"

import numba as nb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if not nb.config.CACHE_DIR:
    nb.config.CACHE_DIR = str(CACHE / "numba")

THETAS = [0.0, 0.0625, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.4375,
          0.5, 0.75]
THETA_LABEL = dict(zip(THETAS, ["0", "1/16", "1/8", "5/32", "3/16", "7/32", "1/4", "9/32", "5/16", "11/32",
                                "3/8", "7/16", "1/2", "3/4"]))
BUDGETS = (5, 10, 15, 20)

REAL = ["ca-GrQc", "Erdos02", "wiki-Vote", "ca-HepTh", "ca-HepPh", "ca-CondMat", "email-Enron",
        "ego-Facebook", "soc-Epinions1", "com-DBLP", "com-Youtube"]
SETTINGS = ["er-d4", "er-d8", "er-d16", "er-d32", "cl-g3.0", "cl-g2.5", "cl-g2.2", "ws-p1.0", "ws-p0.1",
            "ws-p0.01", "collab-s10", "collab-s50", "collab-s200", "er-d8-n50000", "cl-g2.5-n50000"]
GENERATED = [f"{s}-i{i}" for s in SETTINGS for i in range(3)]
NETWORKS = GENERATED + REAL

STRUCTURAL = ["degree", "coverage_greedy", "pagerank", "betweenness"]
ALGORITHMS = STRUCTURAL + ["random"]
SHORT = {"degree": "Deg", "coverage_greedy": "CovG", "pagerank": "PR", "betweenness": "Btw", "random": "Rnd"}
SEQUENCES = ["sequential_a", "sequential_b", "alternating_a", "snake"]
BLOCK = ("sequential_a", "sequential_b")
INTERLEAVED = ("alternating_a", "snake")

SUMMARY_COLUMNS = ["graph_name", "k", "theta", "allocation_order", "algorithm_a", "algorithm_b",
                   "random_involving", "is_self_play", "scientific_realization_count", "mean_a_fraction",
                   "mean_b_fraction", "mean_neutral_fraction", "mean_margin_b", "mean_acquisition_rounds",
                   "mean_ranking_overlap", "mean_skipped_a", "mean_skipped_b", "draw_adjusted_score_a",
                   "draw_adjusted_score_b", "strict_a_win_proportion", "strict_b_win_proportion",
                   "draw_proportion"]


def out_dir(analysis: str) -> Path:
    """Output folder of one analysis, created on first use."""
    path = OUT_ROOT / analysis / "out"
    path.mkdir(parents=True, exist_ok=True)
    return path


def setting_of(network: str) -> str:
    """Generated setting of a network (er-d8-i1 -> er-d8); a real network is its own setting."""
    return re.sub(r"-i\d$", "", network)


# ---------------------------------------------------------------- networks

class Graph(NamedTuple):
    """Simple undirected graph in CSR form, nodes 0..n-1."""
    n: int
    indptr: np.ndarray
    indices: np.ndarray

    @property
    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)


def graph_file(network: str) -> Path:
    """data/<name>.mtx, or the nested layout data/processed|generated/<name>/<name>.mtx."""
    for path in (DATA / f"{network}.mtx", DATA / "processed" / network / f"{network}.mtx",
                 DATA / "generated" / network / f"{network}.mtx"):
        if path.exists():
            return path
    raise FileNotFoundError(f"no Matrix Market file for {network} under {DATA}")


def load_graph(network: str) -> Graph:
    """The network as the engine builds it: 1-based Matrix Market indices made 0-based, every edge
    in both directions, self-loops dropped, duplicate edges merged. Cached under out/cache/graphs."""
    cache = CACHE / "graphs" / f"{network}.npz"
    if cache.exists():
        z = np.load(cache)
        return Graph(int(z["n"]), z["indptr"], z["indices"])
    path = graph_file(network)
    header = 0
    with path.open() as f:
        line = f.readline()
        while line.startswith("%"):
            header += 1
            line = f.readline()
        n = int(line.split()[0])
    edges = pd.read_csv(path, sep=r"\s+", skiprows=header + 1, header=None, usecols=[0, 1],
                        dtype=np.int64, engine="c").to_numpy() - 1
    edges = edges[edges[:, 0] != edges[:, 1]]
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    key = np.unique(src * n + dst)
    src, dst = key // n, key % n
    graph = Graph(n, np.searchsorted(src, np.arange(n + 1)).astype(np.int64), dst.astype(np.int64))
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_name(f"{network}.{os.getpid()}.npz")
    np.savez(partial, n=n, indptr=graph.indptr, indices=graph.indices)
    os.replace(partial, cache)
    return graph


def neighbours_of(graph: Graph, nodes) -> np.ndarray:
    """Concatenated neighbour lists of the given nodes, with repeats."""
    nodes = np.asarray(nodes, dtype=np.int64)
    starts = graph.indptr[nodes]
    lengths = graph.indptr[nodes + 1] - starts
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.repeat(starts - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
    return graph.indices[offsets + np.arange(total)]


# ---------------------------------------------------------------- stored run

def run_dir(network: str, k: int, theta: float) -> Path:
    """Run folder of one network, budget and threshold, e.g. runs/ca-GrQc-k5-t0p1875."""
    tag = "0" if theta == 0 else repr(float(theta)).replace(".", "p")
    return RUN_ROOT / "runs" / f"{network}-k{k}-t{tag}"


def deterministic_matches(network: str, k: int, theta: float = 0.0, self_play: bool = False) -> pd.DataFrame:
    """Run 0 of every match between two structural algorithms (A != B unless self_play) for one
    network and budget, with rankings and seeds parsed. The seeds do not depend on the threshold."""
    columns = ["algorithm_a", "algorithm_b", "run_id", "draft_order", "ranking_a", "ranking_b", "seeds_a",
               "seeds_b"]
    r = pd.read_csv(run_dir(network, k, theta) / "rankings.csv", usecols=columns)
    keep = (r.run_id == 0) & r.algorithm_a.isin(STRUCTURAL) & r.algorithm_b.isin(STRUCTURAL)
    if not self_play:
        keep &= r.algorithm_a != r.algorithm_b
    r = r[keep].drop(columns="run_id").reset_index(drop=True)
    for column in ("ranking_a", "ranking_b", "seeds_a", "seeds_b"):
        r[column] = r[column].map(json.loads)
    return r


def own_top_k(matches: pd.DataFrame) -> dict[str, list[int]]:
    """Each structural algorithm's own top k: its seeds as A under the A-first block sequence."""
    first = matches[matches.draft_order == "sequential_a"].drop_duplicates("algorithm_a")
    return dict(zip(first.algorithm_a, first.seeds_a))


def summaries(table: str = "condition_summaries_cross_play") -> pd.DataFrame:
    """The columns SUMMARY_COLUMNS of one table in <run root>/analysis/combined, in file order,
    cached as a pickle under out/cache/<run root name>/."""
    cache = CACHE / RUN_ROOT.name / f"{table}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    d = pd.read_csv(RUN_ROOT / "analysis" / "combined" / f"{table}.csv", usecols=SUMMARY_COLUMNS)
    d = d[SUMMARY_COLUMNS]
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_name(f"{table}.{os.getpid()}.pkl")
    d.to_pickle(partial)
    os.replace(partial, cache)
    return d


def cross_play() -> pd.DataFrame:
    """Condition summaries of all pairs of different algorithms (Random included)."""
    return summaries("condition_summaries_cross_play")


def self_play() -> pd.DataFrame:
    """Condition summaries of every algorithm against itself."""
    return summaries("self_play_diagnostics")


# ---------------------------------------------------------------- seed allocation and diffusion

def pick_order(k: int, sequence: str) -> list[int]:
    """Who picks at each turn (1 = A, 2 = B) under a picking sequence."""
    if sequence == "sequential_a":
        return [1] * k + [2] * k
    if sequence == "sequential_b":
        return [2] * k + [1] * k
    if sequence == "alternating_a":
        return [1, 2] * k
    if sequence == "snake":
        return [1 if t % 4 in (0, 3) else 2 for t in range(2 * k)]
    raise ValueError(sequence)


def allocate(ranking_a: list[int], ranking_b: list[int], k: int, sequence: str) -> tuple[list[int], list[int]]:
    """Seed allocation: at its turn a player takes the first node of its ranking that nobody owns."""
    taken: set[int] = set()
    seeds = {1: [], 2: []}
    cursor = {1: 0, 2: 0}
    ranking = {1: ranking_a, 2: ranking_b}
    for player in pick_order(k, sequence):
        r = ranking[player]
        while r[cursor[player]] in taken:
            cursor[player] += 1
        node = int(r[cursor[player]])
        cursor[player] += 1
        taken.add(node)
        seeds[player].append(node)
    return seeds[1], seeds[2]


NEUTRAL, PLAYER_A, PLAYER_B = 0, 1, 2
STOP_REASON = {0: "empty_frontier", 1: "no_acquisitions", 2: "max_rounds"}


@nb.njit(cache=True)
def _propagate(indptr, indices, owner, theta, max_rounds):
    """Competitive linear-threshold diffusion in synchronous rounds. A neutral node joins the
    player with strictly more owned neighbours once that count is at least theta * degree; owned
    nodes never change. Only nodes whose counts changed are looked at again."""
    n = owner.shape[0]
    count_a = np.zeros(n, np.int32)
    count_b = np.zeros(n, np.int32)
    in_frontier = np.zeros(n, np.uint8)
    frontier = np.empty(n, np.int64)
    size = 0
    for u in range(n):
        o = owner[u]
        if o == NEUTRAL:
            continue
        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]
            if owner[v] != NEUTRAL:
                continue
            if o == PLAYER_A:
                count_a[v] += 1
            else:
                count_b[v] += 1
            if in_frontier[v] == 0:
                in_frontier[v] = 1
                frontier[size] = v
                size += 1
    changed = np.empty(n, np.int64)
    changed_owner = np.empty(n, np.int8)
    following = np.empty(n, np.int64)
    rounds = 0
    stop = 2
    while rounds < max_rounds:
        if size == 0:
            stop = 0
            break
        n_changed = 0
        for i in range(size):
            u = frontier[i]
            in_frontier[u] = 0
            if owner[u] != NEUTRAL:
                continue
            degree = indptr[u + 1] - indptr[u]
            ca = count_a[u]
            cb = count_b[u]
            if ca > cb and float(ca) >= theta * float(degree):
                changed[n_changed] = u
                changed_owner[n_changed] = PLAYER_A
                n_changed += 1
            elif cb > ca and float(cb) >= theta * float(degree):
                changed[n_changed] = u
                changed_owner[n_changed] = PLAYER_B
                n_changed += 1
        if n_changed == 0:
            stop = 1
            break
        for i in range(n_changed):
            owner[changed[i]] = changed_owner[i]
        next_size = 0
        for i in range(n_changed):
            u = changed[i]
            o = changed_owner[i]
            for p in range(indptr[u], indptr[u + 1]):
                v = indices[p]
                if owner[v] != NEUTRAL:
                    continue
                if o == PLAYER_A:
                    count_a[v] += 1
                else:
                    count_b[v] += 1
                if in_frontier[v] == 0:
                    in_frontier[v] = 1
                    following[next_size] = v
                    next_size += 1
        frontier, following = following, frontier
        size = next_size
        rounds += 1
    return rounds, stop


def propagate(graph: Graph, seeds_a, seeds_b, theta: float, max_rounds: int | None = None):
    """Play out one match from its seeds. Returns the final owner of every node (0 neutral, 1 A,
    2 B), the number of rounds with acquisitions and the stop reason code (see STOP_REASON). With
    no seeds for B, A spreads alone."""
    owner = np.zeros(graph.n, np.int8)
    owner[np.asarray(seeds_a, dtype=np.int64)] = PLAYER_A
    if len(seeds_b):
        owner[np.asarray(seeds_b, dtype=np.int64)] = PLAYER_B
    rounds, stop = _propagate(graph.indptr, graph.indices, owner, float(theta),
                              graph.n if max_rounds is None else max_rounds)
    return owner, rounds, stop


def adoption(graph: Graph, seeds_a, seeds_b, theta: float) -> float:
    """Share of the nodes owned by either player at the end, seeds included."""
    owner, _, _ = propagate(graph, seeds_a, seeds_b, theta)
    return float(np.count_nonzero(owner)) / graph.n
