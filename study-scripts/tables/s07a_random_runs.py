"""Per-run margins of the cross-algorithm conditions with Random.

Reads runs/*/results.csv of the 11 networks: every run's margin (B's nodes minus A's, as a share of
the network) of the conditions with Random. s07_elo uses them for the check variant of the 1% rule.
There a Random condition scores the share of its 100 runs that B wins by more than 1% of the nodes,
plus half the share within +-1%.

Checks that at cutoff 0 these runs reproduce the stored score (win proportion plus half the draw
proportion) of every Random condition, and writes the result to data/random_runs_check.txt.
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import (BUDGETS, CONDITION_KEY, NETWORKS, OUT_DATA, THETAS, batch_dir, cross_play_summaries,
                    output_path)

RESULT_COLUMNS = ["graph", "nodes", "algorithm_a", "algorithm_b", "seed_count", "run_id", "draft_order", "theta",
                  "spread_a", "spread_b"]
RUN_KEY = ["graph", "seed_count", "theta", "draft_order", "algorithm_a", "algorithm_b"]
RUNS_PER_CONDITION = 100
WORKERS = 8


def random_runs_of_batch(results_csv: str) -> pd.DataFrame:
    runs = pd.read_csv(results_csv, usecols=RESULT_COLUMNS)
    with_random = (runs.algorithm_a == "random") | (runs.algorithm_b == "random")
    runs = runs[with_random & (runs.algorithm_a != runs.algorithm_b)]
    runs = runs.assign(margin=(runs.spread_b - runs.spread_a) / runs.nodes,
                       graph=runs.graph.str.replace(".mtx", "", regex=False),
                       theta=runs.theta.astype(float).round(6))
    return runs[RUN_KEY + ["run_id", "margin"]]


def load_random_runs() -> pd.DataFrame:
    paths = sorted(f"{batch_dir(network, k, theta)}/results.csv"
                   for network in NETWORKS for k in BUDGETS for theta in THETAS)
    with Pool(WORKERS) as pool:
        return pd.concat(pool.map(random_runs_of_batch, paths, chunksize=4), ignore_index=True)


def run_scores(runs: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """B's score per Random condition: share of runs with margin > cutoff plus half the share within."""
    score = np.select([runs.margin > cutoff, runs.margin < -cutoff], [1.0, 0.0], 0.5)
    per_condition = (pd.DataFrame({column: runs[column] for column in RUN_KEY}).assign(score=score)
                     .groupby(RUN_KEY).score.agg(["mean", "size"]).reset_index())
    assert (per_condition["size"] == RUNS_PER_CONDITION).all()
    return per_condition.rename(columns={"graph": "graph_name", "seed_count": "k", "draft_order": "allocation_order",
                                         "mean": "score_b"}).drop(columns="size")


def main() -> None:
    runs = load_random_runs()
    print("per-run rows:", len(runs), "conditions:", len(runs) // RUNS_PER_CONDITION)
    summaries = cross_play_summaries()
    with_random = summaries[summaries.random_involving & ~summaries.is_self_play].copy()
    with_random["score_std"] = with_random.strict_b_win_proportion + 0.5 * with_random.draw_proportion
    matched = with_random.merge(run_scores(runs, 0.0), on=CONDITION_KEY, how="outer", validate="one_to_one")
    difference = (matched.score_std - matched.score_b).abs()
    message = (f"Random conditions: summaries {len(with_random)}, per-run groups {matched.score_b.notna().sum()}, "
               f"unmatched {int(matched.score_b.isna().sum() + matched.score_std.isna().sum())}; "
               f"max |stored score - per-run score at cutoff 0| = {difference.max():.3g}")
    print(message)
    with open(output_path(OUT_DATA, "random_runs_check.txt"), "w") as f:
        f.write(message + "\n")


if __name__ == "__main__":
    main()
