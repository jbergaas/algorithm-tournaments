"""Replay the stored matches under every variant of the diffusion model.

A unit is one network, budget (10 or 20) and threshold. For each unit the 64 deterministic matches
of run 0 (16 ordered pairs of the four structural algorithms, self-play included, x 4 sequences,
with their stored seeds) are played under every variant draw of variants.VARIANTS. The base
variant is compared with the stored engine results (spread_a, spread_b, neutral, rounds, stop
reason). For every variant draw except tie, the unit also records each algorithm's first-round
capture: the number of non-seed nodes that its own top k, playing alone, wins in the first round
with the realised node thresholds.

Writes out/units/<network>_k<k>_t<threshold index>.npz (arrays matches: variant, draw, match,
spread_a, spread_b, tied; capture: algorithm, variant, draw, captured) and out/validation.csv.
Usage: python3 diffusion_sensitivity/simulate.py [workers] [networks...]
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import variants as v

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("diffusion_sensitivity")
UNITS = OUT / "units"
LARGEST_FIRST = ("com-Youtube", "com-DBLP", "soc-Epinions1", "email-Enron")
_NETWORK: dict = {}


def network_data(network: str) -> tuple:
    """Graph, seeds per budget and the 20 uniform draws of one network, kept for the next unit."""
    if network not in _NETWORK:
        _NETWORK.clear()
        graph = s.load_graph(network)
        seeds = {}
        for k in v.BUDGETS:
            m = s.deterministic_matches(network, k, self_play=True)
            seeds[k] = {(r.algorithm_a, r.algorithm_b, r.draft_order): (np.asarray(r.seeds_a, np.int64),
                                                                        np.asarray(r.seeds_b, np.int64))
                        for r in m.itertuples()}
            assert set(seeds[k]) == set(v.MATCHES), (network, k)
        draws = [v.uniform_draw(network, graph.n, d) for d in range(v.DRAWS)]
        _NETWORK[network] = (graph, seeds, draws)
    return _NETWORK[network]


def stored_results(network: str, k: int, theta: float) -> pd.DataFrame:
    columns = ["algorithm_a", "algorithm_b", "run_id", "draft_order", "spread_a", "spread_b", "neutral", "rounds",
               "stop_reason"]
    r = pd.read_csv(s.run_dir(network, k, theta) / "results.csv", usecols=columns)
    r = r[(r.run_id == 0) & r.algorithm_a.isin(s.STRUCTURAL) & r.algorithm_b.isin(s.STRUCTURAL)]
    return r.set_index(["algorithm_a", "algorithm_b", "draft_order"])


def own_neighbour_counts(graph: s.Graph, seeds: dict) -> dict:
    """Per algorithm: how many of its own top k neighbour each node (0 for the seeds)."""
    counts = {}
    for alg in s.STRUCTURAL:
        top = seeds[(alg, alg, "sequential_a")][0]
        c = np.bincount(s.neighbours_of(graph, top), minlength=graph.n)
        c[top] = 0
        counts[alg] = c
    return counts


def unit(task: tuple[str, int, int]) -> dict:
    network, k, ti = task
    theta = v.THETAS[ti]
    graph, seeds_by_k, draws = network_data(network)
    seeds = seeds_by_k[k]
    degree = graph.degree.astype(np.float64)
    stored = stored_results(network, k, theta)
    counts = own_neighbour_counts(graph, seeds)
    matches, capture = [], []
    mismatches = 0
    for vi, (name, kind, param, n_draws) in enumerate(v.VARIANTS):
        for d in range(n_draws):
            thr = v.node_thresholds(kind, param, theta, draws[d] if kind in ("unif", "disc") else None, graph.n)
            if kind != "tie":
                for ai, alg in enumerate(s.STRUCTURAL):
                    c = counts[alg]
                    near = c >= 1
                    capture.append((ai, vi, d, int(np.count_nonzero(c[near] >= thr[near] * degree[near]))))
            for mi, (a, b, q) in enumerate(v.MATCHES):
                seeds_a, seeds_b = seeds[(a, b, q)]
                owner = np.zeros(graph.n, np.int8)
                owner[seeds_a] = v.PLAYER_A
                owner[seeds_b] = v.PLAYER_B
                seed = v.match_seed(v.NETWORKS.index(network), k, ti, mi, d, 7919)
                rounds, stop, tied = v.propagate(graph.indptr, graph.indices, owner, thr, kind == "tie", seed,
                                                 graph.n)
                spread_a = int(np.count_nonzero(owner == v.PLAYER_A))
                spread_b = int(np.count_nonzero(owner == v.PLAYER_B))
                matches.append((vi, d, mi, spread_a, spread_b, tied))
                if name == "base":
                    ref = stored.loc[(a, b, q)]
                    mismatches += not (spread_a == ref.spread_a and spread_b == ref.spread_b
                                       and graph.n - spread_a - spread_b == ref.neutral and rounds == ref.rounds
                                       and s.STOP_REASON[stop] == ref.stop_reason)
    np.savez_compressed(UNITS / f"{network}_k{k}_t{ti}.npz", matches=np.asarray(matches, np.int64),
                        capture=np.asarray(capture, np.int64), n=graph.n)
    return dict(network=network, k=k, theta=theta, matches=len(v.MATCHES), mismatches=mismatches)


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    networks = sys.argv[2:] or v.NETWORKS
    UNITS.mkdir(exist_ok=True)
    tasks = [(g, k, ti) for g in networks for k in v.BUDGETS for ti in range(len(v.THETAS))]
    tasks = [t for t in tasks if not (UNITS / f"{t[0]}_k{t[1]}_t{t[2]}.npz").exists()]
    tasks.sort(key=lambda t: (LARGEST_FIRST.index(t[0]) if t[0] in LARGEST_FIRST else len(LARGEST_FIRST), t[0], -t[2]))
    for g in networks:
        s.load_graph(g)
    done = []
    with Pool(workers, maxtasksperchild=40) as pool:
        for row in pool.imap_unordered(unit, tasks, chunksize=1):
            done.append(row)
            print(row, flush=True)
    path = OUT / "validation.csv"
    table = pd.DataFrame(done)
    if path.exists():
        table = pd.concat([pd.read_csv(path), table]).drop_duplicates(["network", "k", "theta"], keep="last")
    table = table.sort_values(["network", "k", "theta"])
    table.to_csv(path, index=False)
    print(f"{table.matches.sum()} base matches compared with the stored results, {table.mismatches.sum()} mismatches")


if __name__ == "__main__":
    main()
