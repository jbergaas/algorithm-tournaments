"""Shared top candidates: how many of their top k two different structural algorithms have in common.

For every network and budget, each structural algorithm's own top k (its first k candidates, read as
its seeds under the A-first block sequence) is compared with that of each other structural
algorithm. The shared share of a pair is |top k of one ∩ top k of the other| / k, for each of the six
unordered pairs; the order within the top k does not matter.

Writes out/shared_top_k.csv (network, k, pair, shared) and out/shared_top_k.txt: per network at
k = 20 the mean, smallest and largest shared share over the six pairs, whether every pair shares at
least one node, and the pairs on the networks named in the text.
Usage: python3 first_pick/shared_top_k.py
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("first_pick")
NAMED = ["Erdos02", "email-Enron", "soc-Epinions1", "com-Youtube", "ca-GrQc", "ca-HepPh"]


def shares() -> pd.DataFrame:
    rows = []
    for network in s.NETWORKS:
        for k in s.BUDGETS:
            top = s.own_top_k(s.deterministic_matches(network, k))
            assert set(top) == set(s.STRUCTURAL), (network, k, sorted(top))
            for x, y in itertools.combinations(s.STRUCTURAL, 2):
                assert len(top[x]) == k and len(top[y]) == k, (network, k, x, y)
                rows.append(dict(network=network, k=k, pair=f"{s.SHORT[x]}/{s.SHORT[y]}",
                                 shared=len(set(top[x]) & set(top[y])) / k))
    return pd.DataFrame(rows)


def main() -> None:
    d = shares()
    d.to_csv(OUT / "shared_top_k.csv", index=False)
    k20 = d[d.k == 20]
    per = (k20.groupby("network").shared.agg(["mean", "min", "max"]) * 100).round(1)
    lines = ["shared top 20 at k = 20, in % of 20, over the six pairs of different structural algorithms",
             "real networks", per.loc[s.REAL].sort_values("mean").to_string(), "",
             "generated networks", per.loc[s.GENERATED].sort_values("mean").to_string(), "",
             f"every pair shares at least one node at k = 20 on all {k20.network.nunique()} networks: "
             f"{bool((k20.shared > 0).all())}", "",
             "per pair at k = 20 (%)",
             (k20[k20.network.isin(NAMED)].pivot(index="network", columns="pair", values="shared") * 100)
             .round(0).loc[NAMED].to_string()]
    text = "\n".join(lines)
    (OUT / "shared_top_k.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
