"""How often the leader of the distance race wins, per threshold, and who leads the race.

Uses every deterministic match between two different structural algorithms (out/race_match.csv
and the engine's mean margin of B). Per threshold, over all networks and budgets: the share of
matches in which race_b and B's margin have the same sign, among those with a race leader and a
nonzero margin (agree_nonzero) and among those decided by more than 1% of the nodes
(agree_decisive). Also the correlation of the two values. Second, the race leader of every network:
the algorithm with the largest mean race share against the other three (over sequences and roles,
then opponents, then budgets).

Writes out/race_by_match.csv and out/race_summary.txt.
Usage: python3 mechanisms/race_summary.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("mechanisms")
KEY = ["graph_name", "k", "algorithm_a", "algorithm_b", "allocation_order"]


def by_threshold(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for theta, y in x.groupby("theta"):
        nonzero = y[(y.mean_margin_b != 0) & (y.race_b != 0)]
        decided = y[(y.mean_margin_b.abs() > 0.01) & (y.race_b != 0)]
        rows.append(dict(theta=s.THETA_LABEL[theta], matches=len(y),
                         agree_nonzero=np.mean(np.sign(nonzero.mean_margin_b) == np.sign(nonzero.race_b)),
                         n_nonzero=len(nonzero),
                         agree_decisive=np.mean(np.sign(decided.mean_margin_b) == np.sign(decided.race_b)),
                         n_decisive=len(decided), pearson=np.corrcoef(y.race_b, y.mean_margin_b)[0, 1]))
    return pd.DataFrame(rows)


def race_leaders(race: pd.DataFrame) -> pd.Series:
    both = pd.concat([race.assign(alg=race.algorithm_b, opp=race.algorithm_a, share=race.race_b),
                      race.assign(alg=race.algorithm_a, opp=race.algorithm_b, share=-race.race_b)])
    share = both.groupby(["graph_name", "k", "alg", "opp"]).share.mean()
    share = share.groupby(["graph_name", "k", "alg"]).mean().groupby(["graph_name", "alg"]).mean().unstack()
    unique = share.apply(lambda r: np.isclose(r, r.max()).sum() == 1, axis=1)
    return share.idxmax(axis=1).where(unique, "tie")


def main() -> None:
    d = s.cross_play()
    d = d[~d.random_involving]
    race = pd.read_csv(OUT / "race_match.csv")
    table = by_threshold(d.merge(race, on=KEY))
    table.to_csv(OUT / "race_by_match.csv", index=False)
    leaders = race_leaders(race)
    lines = ["Leader of the distance race wins (decided matches, all networks and budgets):"]
    lines += [f"  theta {r.theta:>5}: {100 * r.agree_decisive:.1f}% of {r.n_decisive}" for r in table.itertuples()]
    for label, nets in (("all 56 networks", s.NETWORKS), ("the 11 real networks", s.REAL)):
        counts = leaders.reindex(nets).map(lambda a: s.SHORT.get(a, a)).value_counts()
        lines.append(f"Race leader on {label}: " + ", ".join(f"{a} {n}" for a, n in counts.items()))
    text = "\n".join(lines) + "\n"
    (OUT / "race_summary.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
