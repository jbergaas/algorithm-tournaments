"""Replay of the stored deterministic matches, with first-round capture and access.

Every stored match between two different structural algorithms (run 0; 56 networks, four
budgets, 14 thresholds, four sequences, 12 ordered pairs: 150,528 matches) is played again: the
seeds are allocated from the stored rankings and the diffusion is run on the network. Seeds, final
counts, rounds and stop reason are compared with the engine's rankings.csv and results.csv.

For k = 10 and 20 each match also records, for each player:
  solo_a1, solo_b1  nodes its seed set owns after round 1 when it plays alone (seeds included):
                    the first-round capture of the seeds it received
  exp_a, exp_b      access: distinct nodes still neutral after round 1 of the match that are next
                    to one of the player's round-1 captures (non-seed nodes it owns after round 1)

Writes out/matches.csv.gz (one row per match) and out/validation.txt.
Usage: python3 capture/matches.py [workers]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("capture")
DETAIL_BUDGETS = (10, 20)
KEY = ["algorithm_a", "algorithm_b", "run_id", "draft_order"]
BIG_FIRST = ("com-Youtube", "com-DBLP", "soc-Epinions1", "er-d8-n50000-i0", "er-d8-n50000-i1",
             "er-d8-n50000-i2", "cl-g2.5-n50000-i0", "cl-g2.5-n50000-i1", "cl-g2.5-n50000-i2", "email-Enron")


def stored_matches(network: str, k: int, theta: float) -> pd.DataFrame:
    """Rankings, seeds and engine results of the deterministic cross-play matches (run 0)."""
    folder = s.run_dir(network, k, theta)
    rankings = pd.read_csv(folder / "rankings.csv", usecols=KEY + ["ranking_a", "ranking_b", "seeds_a", "seeds_b"])
    results = pd.read_csv(folder / "results.csv", usecols=KEY + ["spread_a", "spread_b", "rounds", "stop_reason"])
    m = rankings.merge(results, on=KEY)
    return m[(m.run_id == 0) & m.algorithm_a.isin(s.STRUCTURAL) & m.algorithm_b.isin(s.STRUCTURAL)
             & (m.algorithm_a != m.algorithm_b)]


def access(graph: s.Graph, after_one: np.ndarray, seeds: list[int], player: int) -> int:
    """Distinct neutral nodes next to the player's round-1 captures."""
    captured = after_one == player
    captured[np.asarray(seeds, dtype=np.int64)] = False
    near = s.neighbours_of(graph, np.flatnonzero(captured))
    return int(np.unique(near[after_one[near] == s.NEUTRAL]).size)


def first_round(graph: s.Graph, seeds_a: list[int], seeds_b: list[int], theta: float) -> dict:
    """First-round capture of each seed set alone and access of each player in the match."""
    alone_a = int(np.count_nonzero(s.propagate(graph, seeds_a, [], theta, max_rounds=1)[0]))
    alone_b = int(np.count_nonzero(s.propagate(graph, seeds_b, [], theta, max_rounds=1)[0]))
    after_one = s.propagate(graph, seeds_a, seeds_b, theta, max_rounds=1)[0]
    return dict(solo_a1=alone_a, solo_b1=alone_b, exp_a=access(graph, after_one, seeds_a, s.PLAYER_A),
                exp_b=access(graph, after_one, seeds_b, s.PLAYER_B))


def replay(task: tuple[str, int]) -> tuple[list[dict], int]:
    """All matches of one network and budget; returns the rows and the number of mismatches."""
    network, k = task
    graph = s.load_graph(network)
    rows, mismatches = [], 0
    for theta in s.THETAS:
        for m in stored_matches(network, k, theta).itertuples(index=False):
            ranking_a, ranking_b = json.loads(m.ranking_a), json.loads(m.ranking_b)
            seeds_a, seeds_b = s.allocate(ranking_a, ranking_b, k, m.draft_order)
            owner, rounds, stop = s.propagate(graph, seeds_a, seeds_b, theta)
            row = dict(graph_name=network, n=graph.n, k=k, theta=theta, draft_order=m.draft_order,
                       algorithm_a=m.algorithm_a, algorithm_b=m.algorithm_b,
                       spread_a=int((owner == s.PLAYER_A).sum()), spread_b=int((owner == s.PLAYER_B).sum()),
                       rounds=int(rounds), stop=s.STOP_REASON[stop])
            mismatches += not (seeds_a == json.loads(m.seeds_a) and seeds_b == json.loads(m.seeds_b)
                               and row["spread_a"] == m.spread_a and row["spread_b"] == m.spread_b
                               and row["rounds"] == m.rounds and row["stop"] == m.stop_reason)
            if k in DETAIL_BUDGETS:
                row.update(first_round(graph, seeds_a, seeds_b, theta))
            rows.append(row)
    return rows, mismatches


def main(workers: int) -> None:
    order = sorted(s.NETWORKS, key=lambda g: BIG_FIRST.index(g) if g in BIG_FIRST else len(BIG_FIRST))
    tasks = [(network, k) for network in order for k in sorted(s.BUDGETS, reverse=True)]
    with Pool(workers, maxtasksperchild=4) as pool:
        parts = dict(zip(tasks, pool.map(replay, tasks, chunksize=1)))
    rows = [row for network in s.NETWORKS for k in s.BUDGETS for row in parts[(network, k)][0]]
    mismatches = sum(part[1] for part in parts.values())
    columns = ["graph_name", "n", "k", "theta", "draft_order", "algorithm_a", "algorithm_b", "spread_a",
               "spread_b", "rounds", "stop", "solo_a1", "solo_b1", "exp_a", "exp_b"]
    table = pd.DataFrame(rows)[columns]
    for column in ("solo_a1", "solo_b1", "exp_a", "exp_b"):
        table[column] = table[column].astype("Int64")
    table.to_csv(OUT / "matches.csv.gz", index=False)
    text = (f"{len(table)} stored deterministic matches on {table.graph_name.nunique()} networks played again; "
            f"matches whose seeds, final counts, rounds or stop reason differ from the engine: {mismatches}\n")
    (OUT / "validation.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
