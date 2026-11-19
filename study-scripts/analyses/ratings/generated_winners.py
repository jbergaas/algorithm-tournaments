"""Winners by threshold regime on the generated networks.

Elo ratings per generated network and regime among the four structural algorithms: every condition
between two of them in the regime is one game, B scores its draw-adjusted score, K = 16 from 1500,
averaged over 200 random game orders. Regimes as in games.py, plus all positive thresholds.

Writes out/elo_generated.csv and out/generated_winners.txt: the order per network and regime, the
most frequent winner of each setting, the largest rating difference between the instances of one
setting, the median absolute margin above the collapse, and, on the ring graphs with p = 0.01, how
many of each algorithm's 20 seeds (k = 20) lie next to another of its own seeds.
Usage: python3 ratings/generated_winners.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402
from elo import elo_variants  # noqa: E402
from games import CELL, regimes  # noqa: E402

OUT = s.out_dir("ratings")
ORDER = ["er-d4", "er-d8", "er-d8-n50000", "er-d16", "er-d32", "cl-g3.0", "cl-g2.5", "cl-g2.5-n50000",
         "cl-g2.2", "ws-p1.0", "ws-p0.1", "ws-p0.01", "collab-s10", "collab-s50", "collab-s200"]
REGIMES = ["low", "spread", "above", "all"]


def ratings() -> pd.DataFrame:
    d = s.cross_play()
    d = d.merge(regimes(d)[CELL + ["regime"]], on=CELL)
    generated = d[~d.graph_name.isin(s.REAL)]
    groups = list(generated.groupby(["graph_name", "regime"]))
    groups += [((g, "all"), x[x.theta > 0]) for g, x in generated.groupby("graph_name")]
    rows = []
    for (network, regime), x in groups:
        y = x[x.algorithm_a.isin(s.STRUCTURAL) & x.algorithm_b.isin(s.STRUCTURAL)]
        values = elo_variants(y.algorithm_a.map(s.STRUCTURAL.index).to_numpy(),
                              y.algorithm_b.map(s.STRUCTURAL.index).to_numpy(),
                              y.draw_adjusted_score_b.to_numpy(), orders=200)
        rows += [dict(graph=network, setting=s.setting_of(network), regime=regime, alg=a, elo=v,
                      median_abs_margin=y.mean_margin_b.abs().median()) for a, v in zip(s.STRUCTURAL, values)]
    return pd.DataFrame(rows)


def winner_lines(e: pd.DataFrame) -> list[str]:
    order = {}
    for (network, regime), x in e.groupby(["graph", "regime"]):
        x = x.sort_values("elo", ascending=False)
        order[(network, regime)] = (f"{s.SHORT[x.alg.iloc[0]]}>{s.SHORT[x.alg.iloc[1]]}..{s.SHORT[x.alg.iloc[-1]]} "
                                    f"({x.elo.iloc[0] - x.elo.iloc[1]:.0f})")
    lines = ["first>second..last (lead in rating points)"]
    for setting in ORDER:
        for i in range(3):
            g = f"{setting}-i{i}"
            lines.append(f"{g:18s} " + " | ".join(f"{r}: {order.get((g, r), '-'):22s}" for r in REGIMES))
    return lines


def most_frequent(winners: pd.Series) -> str:
    if winners.empty:
        return "-"
    counts = winners.value_counts()
    leaders = counts[counts == counts.max()].index
    return leaders[0] if len(leaders) == 1 else "tie"


def most_frequent_lines(e: pd.DataFrame) -> list[str]:
    winners = e.loc[e.groupby(["graph", "regime"]).elo.idxmax()]
    lines = ["most frequent winner per setting and regime (winners of the instances)"]
    for setting in ORDER:
        x = winners[winners.setting == setting].sort_values("graph")
        cells = []
        for regime in REGIMES:
            w = x[x.regime == regime].alg.map(s.SHORT)
            cells.append(f"{regime}: {most_frequent(w)} ({'/'.join(w)})")
        lines.append(f"{setting:15s} " + " | ".join(cells))
    return lines


def adjacency_lines() -> list[str]:
    lines = ["ring graphs p=0.01: own seeds (k=20) next to another own seed"]
    for i in range(3):
        network = f"ws-p0.01-i{i}"
        graph = s.load_graph(network)
        own = s.own_top_k(s.deterministic_matches(network, 20))
        counts = []
        for alg in s.STRUCTURAL:
            seeds = set(own[alg])
            next_to_own = sum(1 for v in seeds if seeds & set(s.neighbours_of(graph, [v]).tolist()))
            counts.append(f"{s.SHORT[alg]} {next_to_own}")
        lines.append(f"{network}: " + ", ".join(counts))
    return lines


def main() -> None:
    e = ratings()
    e.to_csv(OUT / "elo_generated.csv", index=False)
    spread = e.groupby(["setting", "regime", "alg"]).elo.agg(lambda v: v.max() - v.min())
    above = e[e.regime == "above"].drop_duplicates("graph")
    lines = winner_lines(e) + [""] + most_frequent_lines(e) + [""]
    lines += ["largest difference between the instances of one setting (rating points)",
              spread.sort_values().tail(6).round(0).to_string(), ""]
    lines += ["median |margin| of the games above the collapse, largest over the instances",
              above.groupby("setting").median_abs_margin.max().round(4).to_string(), ""]
    lines += adjacency_lines()
    text = "\n".join(lines)
    (OUT / "generated_winners.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
