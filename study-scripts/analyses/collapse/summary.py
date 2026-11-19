"""Collapse intervals of the observed tournament and of its stand-ins, and the numbers quoted about
them.

Per network and budget, an interval is given by the index i of its upper end in the threshold grid
(the collapse lies between THETAS[i-1] and THETAS[i]; 14 means no collapse up to 3/4):
  observed       the collapse interval of observed.py
  degree_duel    one Degree-vs-Degree match (degree_duel.py)
  random_seeds   2k random seeds spreading alone (degree_duel.py)
  deterministic  median adoption of the 48 deterministic matches replayed on the network (rewire.py)
  rewired        the same matches on the rewired copy (rewire.py)
Each simulated interval is found as the observed one: the first sampled threshold above 0 where
the adoption falls below one half.

Writes out/intervals.csv and out/summary.txt (also printed).
Usage: python3 collapse/summary.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("collapse")
INDEX = {t: i for i, t in enumerate(s.THETAS)}
NONE = len(s.THETAS)
CLUSTERED_RINGS = ("ws-p0.1", "ws-p0.01")
UNEVEN = ("cl-g3.0", "cl-g2.5", "cl-g2.2")


def label(i: int) -> str:
    if i >= NONE:
        return ">3/4"
    return f"{s.THETA_LABEL[s.THETAS[i - 1]]}-{s.THETA_LABEL[s.THETAS[i]]}"


def first_below_half(adoption: pd.Series) -> int:
    """Index of the first sampled threshold above 0 at which the adoption is below one half."""
    for i, theta in enumerate(s.THETAS[1:], start=1):
        if adoption[theta] < 0.5:
            return i
    return NONE


def intervals() -> pd.DataFrame:
    observed = pd.read_csv(OUT / "observed_collapse.csv").set_index(["graph", "k"])
    graph_only = pd.read_csv(OUT / "graph_only.csv").set_index(["graph", "k"]).sort_index()
    matches = pd.read_csv(OUT / "deterministic_matches.csv.gz")
    medians = matches.groupby(["graph", "k", "theta"])[["original", "rewired"]].median()
    rows = []
    for network in s.NETWORKS:
        for k in s.BUDGETS:
            alone = graph_only.loc[(network, k)].set_index("theta")
            replayed = medians.loc[(network, k)]
            rows.append(dict(graph=network, setting=s.setting_of(network), k=k, real=network in s.REAL,
                             observed=INDEX.get(observed.loc[(network, k), "collapse_hi"], NONE),
                             degree_duel=first_below_half(alone.degree_duel),
                             random_seeds=first_below_half(alone.random_seeds),
                             deterministic=first_below_half(replayed.original),
                             rewired=first_below_half(replayed.rewired)))
    return pd.DataFrame(rows)


def degree_duel_lines(t: pd.DataFrame) -> list[str]:
    off = (t.degree_duel - t.observed).abs()
    k20 = t[t.k == 20]
    off20 = (k20.degree_duel - k20.observed).abs()
    misses = k20[off20 > 1]
    return [
        "One Degree-vs-Degree match per threshold against the observed collapse interval:",
        f"  within one interval in {int((off <= 1).sum())} of {len(t)} cases "
        f"(same interval in {int((off == 0).sum())})",
        f"  real networks at k=20: within one interval on {int((off20[k20.real] <= 1).sum())} of {int(k20.real.sum())}",
        "  more than one interval off at k=20: "
        + ", ".join(f"{r.graph} ({label(r.degree_duel)} vs {label(r.observed)})" for r in misses.itertuples()),
    ]


def budget_lines(t: pd.DataFrame) -> list[str]:
    shift = t.pivot(index="graph", columns="k", values="observed")
    shift = shift[20] - shift[5]
    return ["", "Observed collapse interval from k=5 to k=20:",
            f"  later on {int((shift > 0).sum())} of {len(shift)} networks, earlier on {int((shift < 0).sum())}"]


def rewiring_lines(t: pd.DataFrame) -> list[str]:
    k20 = t[t.k == 20].set_index("graph")
    lines = ["", "Deterministic matches replayed on the network and on its rewired copy, k=20 "
                 "(intervals earlier on the copy):"]
    rings = [n for n in s.GENERATED if s.setting_of(n) in CLUSTERED_RINGS]
    for network in list(s.REAL) + rings:
        r = k20.loc[network]
        lines.append(f"  {network:14s} network {label(r.deterministic):11s} rewired {label(r.rewired):11s} "
                     f"{r.deterministic - r.rewired}")
    return lines


def random_seed_lines(t: pd.DataFrame) -> list[str]:
    k20 = t[t.k == 20].set_index("graph")
    random_graphs = sorted({label(i) for i in k20.loc[[f"er-d8-i{i}" for i in range(3)], "observed"]})
    uneven = k20[k20.setting.isin(UNEVEN)]
    same = uneven.random_seeds.map(label).isin(random_graphs)
    return ["", "2k random seeds spreading alone on the uneven graphs, k=20:",
            f"  er-d8 collapses at {', '.join(random_graphs)}; the random seeds collapse there on "
            f"{int(same.sum())} of {len(uneven)} uneven graphs",
            "  " + ", ".join(f"{g} {label(i)}" for g, i in uneven.random_seeds.items())]


def main() -> None:
    t = intervals()
    t.to_csv(OUT / "intervals.csv", index=False)
    lines = degree_duel_lines(t) + budget_lines(t) + rewiring_lines(t) + random_seed_lines(t)
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
