"""Shared top candidates.

Writes:
  data/similarity.csv      per network and budget k, the share (%) of the top k that each pair of
                           deterministic algorithms has in common, read from the recorded rankings
                           of runs/<network>-k<k>-t0/rankings.csv (rankings do not depend on theta)
  data/random_overlap.csv  per network and budget, the Random conditions whose top k shares at least
                           one node with the opponent's (h_k > 0), observed and expected under a
                           hypergeometric model, out of 800 (eight ordered pairs x 100 runs)
"""
import csv
import itertools
import json

import pandas as pd

from common import (BUDGETS, LETTER, NETWORKS, NODES, OUT_DATA, STRUCTURAL, batch_dir, output_path,
                    overlap_conditions)

csv.field_size_limit(10**9)
RANDOM_USES = 800


def chance_of_shared_node(n: int, k: int) -> float:
    """Probability that k nodes drawn at random from n include one of k given nodes."""
    p_none = 1.0
    for i in range(k):
        p_none *= (n - k - i) / (n - i)
    return 1 - p_none


def top_k_rankings(network: str, k: int) -> dict:
    """First k entries of each structural algorithm's ranking."""
    top = {}
    with open(f"{batch_dir(network, k, 0)}/rankings.csv") as f:
        for row in csv.DictReader(f):
            for side in ("a", "b"):
                algorithm = row[f"algorithm_{side}"]
                if algorithm in STRUCTURAL and algorithm not in top:
                    top[algorithm] = json.loads(row[f"ranking_{side}"])[:k]
            if len(top) == len(STRUCTURAL):
                break
    return top


def similarity_table() -> pd.DataFrame:
    pairs = list(itertools.combinations(STRUCTURAL, 2))
    rows = []
    for k in BUDGETS:
        for network in NETWORKS:
            top = top_k_rankings(network, k)
            shared = {(a, b): len(set(top[a]) & set(top[b])) for a, b in pairs}
            shares = {f"{LETTER[a]}-{LETTER[b]}": 100 * shared[(a, b)] / k for a, b in pairs}
            rows.append(dict(network=network, k=k, **shares, mean=sum(shares.values()) / len(shares),
                             min_shared_nodes=min(shared.values())))
    return pd.DataFrame(rows)


def random_overlap_table() -> pd.DataFrame:
    with_random = overlap_conditions()
    with_random = with_random[with_random.random_involving]
    per_setting = (with_random.groupby(["graph_name", "k", "theta", "allocation_order"])
                   .top_k_overlap_eligible_count.sum())
    spread = per_setting.groupby(level=["graph_name", "k"]).agg(["min", "max"])
    print("Random overlap invariant over the 14 thresholds and 4 sequences:",
          bool((spread["min"] == spread["max"]).all()))

    # The count is the same at every threshold and sequence (checked above), so one setting suffices.
    one_setting = with_random[(with_random.theta == 0) & (with_random.allocation_order == "sequential_a")]
    observed = one_setting.groupby(["graph_name", "k"]).top_k_overlap_eligible_count.sum()
    uses = one_setting.groupby(["graph_name", "k"]).scientific_realization_count.sum()
    rows = []
    for network in NETWORKS:
        row = dict(network=network, n=NODES[network])
        for k in BUDGETS:
            row[f"obs_k{k}"] = int(observed[(network, k)])
            row[f"uses_k{k}"] = int(uses[(network, k)])
            row[f"exp_k{k}"] = RANDOM_USES * chance_of_shared_node(NODES[network], k)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("network")
    assert (table[[f"uses_k{k}" for k in BUDGETS]] == RANDOM_USES).all().all()
    return table


def main() -> None:
    similarity = similarity_table()
    similarity.to_csv(output_path(OUT_DATA, "similarity.csv"), index=False)
    overlap = random_overlap_table()
    overlap.to_csv(output_path(OUT_DATA, "random_overlap.csv"))

    pd.set_option("display.width", 250)
    print(similarity[similarity.k == 20].set_index("network").round(1).to_string())
    print(overlap[["obs_k5", "exp_k5", "obs_k20", "exp_k20"]].round(1).to_string())
    print(f"k=20 expected {overlap.exp_k20.sum():.1f}, observed {overlap.obs_k20.sum()}; "
          f"k=5 expected {overlap.exp_k5.sum():.1f}, observed {overlap.obs_k5.sum()}")


if __name__ == "__main__":
    main()
