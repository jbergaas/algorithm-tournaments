"""The three winner predictors on the same games, overall, by picking sequence and by threshold.

Games (predictors/games.py): deterministic matches between two different structural algorithms
at k = 10 and 20 on the real networks and the first instance of each generated setting, with
1/16 < theta < collapse (collapse/out/observed_collapse.csv), decided (|margin| > 1% of the
nodes) and with neither player level on any of the three predictors. Each predictor is measured
on the seeds the player received in that game:
  race     more nodes nearer to its seeds (mechanisms/out/race_match.csv)
  capture  more nodes its seeds alone would own after round 1 (solo_a1, solo_b1)
  access   more still-neutral nodes next to its round-1 captures in the match (exp_a, exp_b)
For comparison, capture measured on each algorithm's own top k (its seeds as A under the A-first
block sequence), which does not depend on the sequence (left empty when level).

Writes out/same_games.csv.gz (one row per game, a hit flag per predictor; input of
figures/predictors.py) and out/same_games.txt.
Usage: python3 predictors/same_games.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import games  # noqa: E402
import shared as s  # noqa: E402

OUT = s.out_dir("predictors")
HITS = ["race_hit", "capture_hit", "access_hit"]


def in_band(m: pd.DataFrame) -> pd.DataFrame:
    """Thresholds above 1/16 and below the budget's collapse."""
    collapse = pd.read_csv(s.out_dir("collapse") / "observed_collapse.csv", usecols=["graph", "k", "collapse_hi"])
    m = m.merge(collapse.rename(columns={"graph": "graph_name", "collapse_hi": "collapse"}), on=["graph_name", "k"])
    return m[(m.theta > 1 / 16 + 1e-9) & (m.theta < m.collapse - 1e-9)]


def with_own_capture(m: pd.DataFrame) -> pd.DataFrame:
    own = (m[m.draft_order == "sequential_a"][["graph_name", "k", "theta", "algorithm_a", "solo_a1"]]
           .drop_duplicates(["graph_name", "k", "theta", "algorithm_a"]))
    for side in ("a", "b"):
        m = m.merge(own.rename(columns={"algorithm_a": f"algorithm_{side}", "solo_a1": f"own_{side}"}),
                    on=["graph_name", "k", "theta", f"algorithm_{side}"], how="left")
    return m


def hits(m: pd.DataFrame) -> pd.DataFrame:
    winner_b = np.sign(m.margin_b)
    leads = {"race": np.sign(m.race_b), "capture": np.sign(m.solo_b1 - m.solo_a1),
             "access": np.sign(m.exp_b - m.exp_a)}
    keep = (m.margin_b.abs() > 0.01) & np.logical_and.reduce([lead != 0 for lead in leads.values()])
    own = np.sign(m.own_b - m.own_a)
    f = pd.DataFrame({"graph_name": m.graph_name, "k": m.k, "theta": m.theta, "sequence": m.draft_order,
                      **{f"{p}_hit": lead == winner_b for p, lead in leads.items()},
                      "own_capture_hit": np.where(own != 0, own == winner_b, np.nan)})
    return f[keep].reset_index(drop=True)


def summary(f: pd.DataFrame) -> list[str]:
    lines = [f"games: {len(f)} on {f.graph_name.nunique()} networks (decided, none of the three level)",
             "share won by the player ahead (%):"]
    group = np.where(f.sequence.isin(s.BLOCK), "block", "interleaved")
    for label, g in [("all", f)] + list(f.groupby(group)) + list(f.groupby("sequence")):
        own = g.own_capture_hit.dropna()
        lines.append(f"  {label:14s} n={len(g):5d}  " + "  ".join(f"{c[:-4]} {100 * g[c].mean():5.1f}" for c in HITS)
                     + f"  own-top-k capture {100 * own.mean():5.1f} (n={len(own)})")
    lines.append("by threshold, block | interleaved:")
    for theta, g in f.groupby("theta"):
        parts = []
        for name, h in g.groupby(np.where(g.sequence.isin(s.BLOCK), "block", "interleaved")):
            parts.append(f"{name} n={len(h):4d} " + " ".join(f"{100 * h[c].mean():5.1f}" for c in HITS))
        lines.append(f"  {theta:.5f}  " + " | ".join(parts))
    return lines


def main() -> None:
    f = hits(with_own_capture(in_band(games.load())))
    f.to_csv(OUT / "same_games.csv.gz", index=False)
    text = "\n".join(summary(f)) + "\n"
    (OUT / "same_games.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
