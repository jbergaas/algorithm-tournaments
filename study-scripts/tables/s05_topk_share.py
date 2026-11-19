"""Who receives its own top candidates, and does that player win (tab:results-topk-share).

Reads results.csv and rankings.csv of every batch of the 11 networks in runs/ for run 0 of each
deterministic cross-algorithm condition: the share of its own top k that each player receives as
seeds, the overlap of the two top-k lists and B's margin. Writes data/topk_share.csv and prints the
numbers quoted with it. Per network at positive thresholds, the CSV holds the mean overlap, the
settings where the shares differ, and how often the player with the larger share wins.
"""
import csv
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import (BUDGETS, NETWORKS, OUT_DATA, THETA_FRACTION, THETAS, batch_dir, is_deterministic_cross_play,
                    output_path)

csv.field_size_limit(10**9)
WORKERS = 6


def batch_conditions(folder: str) -> list:
    """One row per deterministic cross-algorithm condition of a batch (run 0)."""
    results = {}
    with open(os.path.join(folder, "results.csv")) as f:
        for row in csv.DictReader(f):
            if is_deterministic_cross_play(row):
                results[(row["algorithm_a"], row["algorithm_b"], row["draft_order"])] = row
    conditions = []
    with open(os.path.join(folder, "rankings.csv")) as f:
        for row in csv.DictReader(f):
            key = (row["algorithm_a"], row["algorithm_b"], row["draft_order"])
            if row["run_id"] != "0" or key not in results:
                continue
            k = int(row["seed_count"])
            top_a = set(json.loads(row["ranking_a"])[:k])
            top_b = set(json.loads(row["ranking_b"])[:k])
            seeds_a, seeds_b = set(json.loads(row["seeds_a"])), set(json.loads(row["seeds_b"]))
            result = results[key]
            n = int(result["nodes"])
            conditions.append(dict(graph=result["graph"].replace(".mtx", ""), k=k,
                                   theta=round(float(result["theta"]), 6), order=key[2], alg_a=key[0],
                                   alg_b=key[1], share_a=len(seeds_a & top_a) / k, share_b=len(seeds_b & top_b) / k,
                                   overlap=len(top_a & top_b) / k,
                                   m_b=(int(result["spread_b"]) - int(result["spread_a"])) / n))
    return conditions


def load_conditions() -> pd.DataFrame:
    batches = sorted(batch_dir(network, k, theta) for network in NETWORKS for k in BUDGETS for theta in THETAS)
    print("batches:", len(batches), flush=True)
    with Pool(WORKERS) as pool:
        per_batch = pool.map(batch_conditions, batches, chunksize=4)
    return pd.DataFrame([row for rows in per_batch for row in rows])


def larger_share_wins(conditions: pd.DataFrame) -> float:
    """Percentage of conditions won by the player with the larger share (a draw counts as neither)."""
    return 100 * (np.sign(conditions.share_lead_b) == np.sign(conditions.m_b)).mean()


def summarise(conditions: pd.DataFrame) -> None:
    positive = conditions[conditions.theta > 0].copy()
    positive["share_lead_b"] = positive.share_b - positive.share_a
    contested = positive[positive.share_lead_b != 0]
    print(f"\npositive-threshold conditions {len(positive)}; contested {len(contested)}; "
          f"larger share wins {larger_share_wins(contested):.1f}%")
    for sequence, group in positive.groupby("order"):
        group_contested = group[group.share_lead_b != 0]
        print(f"  {sequence:14s} mean share A {100*group.share_a.mean():5.1f}%  B {100*group.share_b.mean():5.1f}%  "
              f"larger share wins {larger_share_wins(group_contested):5.1f}% of {len(group_contested)}")
    block = contested[contested.order.isin(["sequential_a", "sequential_b"])]
    print(f"  block sequences pooled: larger share (= first picker) wins {larger_share_wins(block):.1f}% "
          f"of {len(block)}")

    rows = []
    for network, group in positive.groupby("graph"):
        group_contested = group[group.share_lead_b != 0]
        rows.append(dict(network=network, shared_pct=100 * group.overlap.mean(), settings=len(group_contested),
                         larger_wins_pct=larger_share_wins(group_contested)))
    table = pd.DataFrame(rows).set_index("network").loc[NETWORKS]
    table.to_csv(output_path(OUT_DATA, "topk_share.csv"))
    print(table.round(1).to_string())

    def contested_rate(group: pd.DataFrame) -> float:
        return larger_share_wins(group[group.share_lead_b != 0])

    print("  by theta:", {THETA_FRACTION[t]: round(contested_rate(g), 1) for t, g in positive.groupby("theta")})
    print("  by k:", {k: round(contested_rate(g), 1) for k, g in positive.groupby("k")})
    print(f"  thresholds up to 3/16: {contested_rate(positive[positive.theta <= 0.1875]):.1f}%; at 3/4: "
          f"{contested_rate(positive[positive.theta == 0.75]):.1f}%")


def main() -> None:
    conditions = load_conditions()
    print("conditions:", len(conditions), conditions.groupby("graph").size().to_dict())
    summarise(conditions)


if __name__ == "__main__":
    main()
