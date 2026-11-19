"""Collapse interval of every network and budget.

The median adoption of a network, budget and threshold is the median share of nodes owned by
either player at the end, over the 80 cross-algorithm conditions (20 ordered pairs of different
algorithms, four sequences). A condition with Random counts as the mean of its 100 runs. The spread
collapses at the first sampled threshold above 0 where the median adoption falls below one half.
The collapse interval runs from the threshold before it to that threshold.

Writes out/observed_collapse.csv (graph, k, collapse_lo, collapse_hi, median_lo, median_hi; empty
when the median never falls below one half) and prints how the interval moves from k = 5 to k = 20.
Usage: python3 collapse/observed.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("collapse")


def median_adoption(d: pd.DataFrame) -> pd.DataFrame:
    d = d.assign(adoption=1 - d.mean_neutral_fraction)
    g = d.groupby(["graph_name", "k", "theta"]).adoption
    out = pd.DataFrame({"conditions": g.size(), "median_adoption": g.median()}).reset_index()
    assert (out.conditions == 80).all(), out.conditions.unique()
    return out


def collapse_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (graph, k), x in curves.groupby(["graph_name", "k"]):
        median = x.set_index("theta").median_adoption.reindex(s.THETAS).to_numpy()
        below = [i for i in range(1, len(s.THETAS)) if median[i] < 0.5]
        row = dict(graph=graph, k=k, collapse_lo=np.nan, collapse_hi=np.nan, median_lo=np.nan, median_hi=np.nan)
        if below:
            i = below[0]
            row.update(collapse_lo=s.THETAS[i - 1], collapse_hi=s.THETAS[i], median_lo=median[i - 1],
                       median_hi=median[i])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    curves = median_adoption(s.cross_play())
    table = collapse_table(curves)
    table.to_csv(OUT / "observed_collapse.csv", index=False)
    index = table.assign(i=table.collapse_hi.map({t: i for i, t in enumerate(s.THETAS)}).fillna(len(s.THETAS)))
    shift = index.pivot(index="graph", columns="k", values="i")
    shift = shift[20] - shift[5]
    print(f"collapse interval from k=5 to k=20: later on {int((shift > 0).sum())} of {len(shift)} networks, "
          f"earlier on {int((shift < 0).sum())}, same on {int((shift == 0).sum())}")


if __name__ == "__main__":
    main()
