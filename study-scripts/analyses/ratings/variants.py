"""The 260 variants of the ratings on the real networks, with all four picking sequences and with
strict alternation and snake alone.

A variant combines four choices: the threshold grid (all 14 thresholds, or the 10 without 5/32,
7/32, 9/32 and 11/32), one threshold of that grid left out or none (15 + 11 sets), the cutoff for a
win (a margin of more than 0, 0.25, 0.5, 1 or 2% of the nodes, otherwise a draw) and the rating (Elo
over 100 random game orders, or the mean score). Each variant rates the four structural algorithms
on every real network in each regime (low, spread, above) and over all thresholds (overall).

Writes out/variants.csv.gz (one row per sequence set, network, regime and variant) and
out/variants_summary.txt with, per sequence set and regime, the algorithm with the highest mean over
the eleven networks and the number of networks on which each algorithm is first alone.
Usage: python3 ratings/variants.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402
from elo import elo_variants, mean_scores  # noqa: E402
from games import structural_games  # noqa: E402

OUT = s.out_dir("ratings")
GRIDS = {"exp": s.THETAS, "orig": [t for t in s.THETAS if t not in (0.15625, 0.21875, 0.28125, 0.34375)]}
CUTOFFS = [0.0, 0.0025, 0.005, 0.01, 0.02]
REGIMES = ["low", "spread", "above", "overall"]
VARIANT = ["grid", "drop", "cutoff", "score"]
SEQUENCE_SETS = {"all": tuple(s.SEQUENCES), "interleaved": s.INTERLEAVED}
SHORT = [s.SHORT[a] for a in s.STRUCTURAL]


def score_b(margin_b: np.ndarray, cutoff: float) -> np.ndarray:
    """B's score when a game is won only by a margin above the cutoff and is otherwise a draw."""
    return np.select([margin_b > cutoff, margin_b < -cutoff], [1.0, 0.0], 0.5)


def ratings(games: pd.DataFrame) -> list[dict]:
    """One row per regime and variant: the four values and the algorithms first alone."""
    a = games.algorithm_a.map(s.STRUCTURAL.index).to_numpy()
    b = games.algorithm_b.map(s.STRUCTURAL.index).to_numpy()
    theta, regime = games.theta.to_numpy(), games.regime.to_numpy()
    scores = {cut: score_b(games.mean_margin_b.to_numpy(), cut) for cut in CUTOFFS}
    rows = []
    for grid, thetas in GRIDS.items():
        for drop in [None] + thetas:
            kept = np.isin(theta, [t for t in thetas if t != drop])
            for reg in REGIMES:
                mask = kept if reg == "overall" else kept & (regime == reg)
                if not mask.any():
                    continue
                for cut in CUTOFFS:
                    for score in ("elo", "mean"):
                        if score == "elo":
                            values, tol = elo_variants(a[mask], b[mask], scores[cut][mask]), 1e-6
                        else:
                            values, tol = mean_scores(a[mask], b[mask], scores[cut][mask]), 1e-9
                        first = [i for i in range(4) if abs(values[i] - values.max()) <= tol]
                        row = dict(regime=reg, grid=grid, drop="none" if drop is None else s.THETA_LABEL[drop],
                                   cutoff=cut, score=score)
                        row.update(zip(SHORT, values))
                        row.update({f"first_{name}": first == [i] for i, name in enumerate(SHORT)})
                        rows.append(row)
    return rows


def summary(v: pd.DataFrame) -> list[str]:
    lines = []
    for sequences, x in v.groupby("sequences", sort=False):
        lines.append(f"{sequences} sequences: {x.groupby(VARIANT).ngroups} variants")
        for reg in REGIMES:
            per_variant = x[x.regime == reg].groupby(VARIANT)
            top = per_variant[SHORT].mean().idxmax(axis=1).value_counts().to_dict()
            lines.append(f"  {reg:8s} highest mean over the 11 networks -> variants: {top}")
            for name in SHORT:
                counts = per_variant[f"first_{name}"].sum().value_counts().sort_index().to_dict()
                lines.append(f"           networks with {name} first alone -> variants: {counts}")
    return lines


def main() -> None:
    games = structural_games()
    games = games[games.graph_name.isin(s.REAL)]
    rows = []
    for name, sequences in SEQUENCE_SETS.items():
        subset = games[games.allocation_order.isin(sequences)]
        for network, x in subset.groupby("graph_name", sort=False):
            rows += [dict(sequences=name, graph=network, **r) for r in ratings(x)]
        print(name, "done", flush=True)
    v = pd.DataFrame(rows)
    v.to_csv(OUT / "variants.csv.gz", index=False)
    text = "\n".join(summary(v))
    (OUT / "variants_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
