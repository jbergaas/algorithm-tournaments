"""Elo ratings of the result table by threshold regime, recomputed on strict alternation and snake
alone, without the two block sequences.

The method is that of the table. Every cross-algorithm condition of a real network (Random
included) in the regime is one game, and B scores its win proportion plus half its draw proportion.
Elo with K = 16 from 1500 is averaged over 200 random game orders, drawn from one generator (seed
20261104) per regime, network by network in the table's order. Regimes: R1 (theta <= 1/16), R2
(spreading), R3 (post-collapse) and ALL (every positive threshold). With all four sequences the
ratings equal those of the tables; this is checked against data/elo_all.csv of ../tables when that
file is present.

Writes out/elo_interleaved.csv and out/interleaved_summary.txt: for all sequences and for the
interleaved ones, the mean rating per algorithm over the networks (rounded, as in the table), the lead
of the highest mean and the first-placed algorithm per network.
Usage: python3 ratings/interleaved.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402
from elo import elo_table  # noqa: E402
from games import CELL, regimes  # noqa: E402

OUT = s.out_dir("ratings")
SEED = 20261104
TABLE_ORDER = ["Erdos02", "ca-GrQc", "wiki-Vote", "ca-HepPh", "ca-CondMat", "email-Enron", "com-DBLP",
               "com-Youtube", "ca-HepTh", "soc-Epinions1", "ego-Facebook"]
LABEL = {"low": "R1", "spread": "R2", "above": "R3"}
REGIMES = ["R1", "R2", "R3", "ALL"]
SEQUENCE_SETS = {"all": tuple(s.SEQUENCES), "interleaved": s.INTERLEAVED}


def games() -> pd.DataFrame:
    """Cross-algorithm conditions of the real networks with B's standard score and the regime."""
    d = s.cross_play()
    reg = regimes(d)
    reg["regime"] = reg.regime.map(LABEL)
    g = d[d.graph_name.isin(s.REAL) & ~d.is_self_play].copy()
    g["score"] = g.strict_b_win_proportion + 0.5 * g.draw_proportion
    return g.merge(reg[CELL + ["regime"]], on=CELL, how="left", validate="many_to_one")


def rate(g: pd.DataFrame, regime: str) -> list[dict]:
    rng = np.random.default_rng(SEED)
    x = g[g.theta > 0] if regime == "ALL" else g[g.regime == regime]
    rows = []
    for network in TABLE_ORDER:
        y = x[x.graph_name == network]
        values = elo_table(y.algorithm_a.map(s.ALGORITHMS.index).to_numpy(),
                           y.algorithm_b.map(s.ALGORITHMS.index).to_numpy(), y.score.to_numpy(dtype=float), rng)
        rows += [dict(regime=regime, network=network, alg=a, elo=v, games=len(y)) for a, v in zip(s.ALGORITHMS, values)]
    return rows


def check_tables(ratings: pd.DataFrame) -> str:
    path = s.TABLES_DATA / "elo_all.csv"
    if not path.exists():
        return "data/elo_all.csv of ../tables not found; check skipped"
    ref = pd.read_csv(path)
    ref = ref[(ref.scope_name == "ext") & (ref.cutoff == "std") & (ref.seed == SEED) & ref.regime.isin(REGIMES)]
    mine = ratings[ratings.sequences == "all"].merge(ref, on=["regime", "network", "alg"], suffixes=("", "_ref"))
    assert len(mine) == len(ref) == 11 * 5 * len(REGIMES), (len(mine), len(ref))
    worst = float((mine.elo - mine.elo_ref).abs().max())
    assert worst < 1e-6, worst
    return f"all sequences equal the tables' ratings (largest difference {worst:.1e})"


def summary(ratings: pd.DataFrame) -> list[str]:
    lines = []
    for regime in REGIMES:
        lines.append(f"regime {regime}")
        for sequences, x in ratings[ratings.regime == regime].groupby("sequences", sort=False):
            e = x.pivot(index="network", columns="alg", values="elo").loc[TABLE_ORDER, s.ALGORITHMS]
            mean = e.mean().round()
            ranked = mean[s.STRUCTURAL].sort_values(ascending=False)
            first = e[s.STRUCTURAL].idxmax(axis=1)
            lines.append(f"  {sequences:11s} mean " + " ".join(f"{s.SHORT[a]} {mean[a]:.0f}" for a in s.ALGORITHMS)
                         + f"; highest {s.SHORT[ranked.index[0]]}, {ranked.iloc[0] - ranked.iloc[1]:.0f} ahead"
                         + "; first on: " + " ".join(f"{s.SHORT[a]} {int((first == a).sum())}" for a in s.STRUCTURAL))
            lines.append("    first per network: " + ", ".join(f"{n} {s.SHORT[first[n]]}" for n in TABLE_ORDER))
    return lines


def main() -> None:
    g = games()
    rows = []
    for sequences, allowed in SEQUENCE_SETS.items():
        subset = g[g.allocation_order.isin(allowed)]
        rows += [dict(sequences=sequences, **r) for regime in REGIMES for r in rate(subset, regime)]
    ratings = pd.DataFrame(rows)
    ratings.to_csv(OUT / "elo_interleaved.csv", index=False)
    text = "\n".join([check_tables(ratings)] + summary(ratings))
    (OUT / "interleaved_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
