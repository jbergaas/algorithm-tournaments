"""What knowing more is worth to a player that may choose one of the four structural algorithms.

The games are the deterministic matches between structural algorithms, each seen from both sides.
The opponent uses each of the four algorithms equally often; when it uses the player's own choice,
the recorded self-play match counts (over the two roles it averages to a draw). A game scores 1 for
a win, 1/2 for a draw and 0 for a loss. Every network unit (a real network, or a generated setting
with its three instances) weighs 1, shared equally by its budget and threshold cells.

At each level of knowledge the player takes the algorithm with the highest score over the
situations it cannot tell apart (ties: the first in alphabetical order). The levels are the rows of
the table: always PageRank, the network and budget, plus the threshold, plus the opponent, plus the
picking sequence and role, all three, and the opponent with the sequence and role but without the
threshold. Every level is also scored with the own pairing counted as a
draw, which gives the same values at the levels that do not know the role.

Writes out/levels.csv and out/knowledge.txt: per level the score, wins and draws (real and
generated networks); the second picker's score under a block sequence when all three are known; the
choice per budget and the gain over PageRank on each real network without the threshold; and how
often the algorithm best against all four opponents is also the best reply to a single opponent.
Usage: python3 knowledge/value_of_knowledge.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402
from ratings.games import CELL, structural_games, unit_of  # noqa: E402

OUT = s.out_dir("knowledge")
ALGS = ["Deg", "CovG", "PR", "Btw"]
LEVELS = {
    "always PageRank": None,
    "network and budget": ["graph_name", "k"],
    "threshold": CELL,
    "threshold + opponent": CELL + ["opp"],
    "threshold + sequence and role": CELL + ["seq", "role"],
    "all three": CELL + ["opp", "seq", "role"],
    "opponent, sequence and role, no threshold": ["graph_name", "k", "opp", "seq", "role"],
}
ROLE_LEVELS = ("threshold + sequence and role", "all three", "opponent, sequence and role, no threshold")


def player_rows(d: pd.DataFrame) -> pd.DataFrame:
    """Each match from A's side and from B's side: player, opponent, sequence, role, score, margin."""
    a, b = d.algorithm_a.map(s.SHORT), d.algorithm_b.map(s.SHORT)
    cell = {c: d[c] for c in CELL}
    as_a = pd.DataFrame({**cell, "seq": d.allocation_order, "player": a, "opp": b, "role": "A",
                         "score": 1 - d.draw_adjusted_score_b, "margin": -d.mean_margin_b})
    as_b = pd.DataFrame({**cell, "seq": d.allocation_order, "player": b, "opp": a, "role": "B",
                         "score": d.draw_adjusted_score_b, "margin": d.mean_margin_b})
    return pd.concat([as_a, as_b], ignore_index=True)


def own_pairing(cells: pd.DataFrame, recorded: bool) -> pd.DataFrame:
    """The player against its own algorithm: the recorded self-play match, or a draw."""
    if not recorded:
        return pd.concat([cells.assign(player=a, opp=a, score=0.5, margin=0.0) for a in ALGS], ignore_index=True)
    sp = s.self_play()
    sp = sp[sp.is_self_play & sp.algorithm_a.isin(s.STRUCTURAL)].copy()
    assert (sp.algorithm_a == sp.algorithm_b).all()
    sp["theta"] = sp.theta.round(9)
    rows = player_rows(sp)
    out = rows.merge(cells[CELL].drop_duplicates(), on=CELL, how="inner")
    assert len(out) == cells[CELL].drop_duplicates().shape[0] * len(ALGS) * 4 * 2
    return out


def fine_table(recorded: bool) -> pd.DataFrame:
    """One row per cell, player, opponent, sequence and role, with the weight of the row."""
    g = player_rows(structural_games())
    g["theta"] = g.theta.round(9)
    if recorded:
        f = pd.concat([g, own_pairing(g.drop_duplicates(CELL)[CELL], True)], ignore_index=True)
    else:
        cells = g.drop_duplicates(CELL + ["seq", "role"])[CELL + ["seq", "role"]]
        f = pd.concat([g, own_pairing(cells, False)], ignore_index=True)
    f["unit"] = f.graph_name.map(unit_of)
    f["side"] = np.where(f.graph_name.isin(s.REAL), "real", "generated")
    cells_per_unit = f.drop_duplicates(CELL).groupby("unit").size()
    f["w"] = 1.0 / f.unit.map(cells_per_unit) / 32.0
    return f


def level_value(f: pd.DataFrame, keys: list[str] | None) -> float:
    if keys is None:
        x = f[f.player == "PR"]
        return (x.w * x.score).sum() / (f.w.sum() / len(ALGS))
    total = f.assign(ws=f.w * f.score).groupby(["player"] + keys).ws.sum()
    return total.groupby(level=keys).max().sum() / (f.w.sum() / len(ALGS))


def chosen_rows(f: pd.DataFrame, keys: list[str] | None) -> pd.DataFrame:
    """The rows of the algorithm the player takes at this level."""
    if keys is None:
        return f[f.player == "PR"]
    total = f.assign(ws=f.w * f.score).groupby(["player"] + keys).ws.sum().reset_index()
    best = total.loc[total.groupby(keys).ws.idxmax()]
    return f.merge(best[["player"] + keys], on=["player"] + keys)


def levels(recorded: pd.DataFrame, draw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ("real", "generated"):
        f, f_draw = recorded[recorded.side == side], draw[draw.side == side]
        weight = f.w.sum() / len(ALGS)
        for name, keys in LEVELS.items():
            chosen = chosen_rows(f, keys)
            rows.append(dict(side=side, level=name, score=100 * level_value(f, keys),
                             draw_convention=100 * level_value(f_draw, keys),
                             wins=100 * chosen[chosen.score > 0.5].w.sum() / weight,
                             draws=100 * chosen[chosen.score == 0.5].w.sum() / weight))
    t = pd.DataFrame(rows)
    no_role = ~t.level.isin(ROLE_LEVELS)
    assert ((t.score - t.draw_convention)[no_role].abs() < 1e-7).all()
    return t


def second_picker_lines(recorded: pd.DataFrame) -> list[str]:
    chosen = chosen_rows(recorded[recorded.side == "real"], LEVELS["all three"])
    lines = ["Real networks, all three known: score by sequence and role (%)"]
    for (seq, role), x in chosen.groupby(["seq", "role"]):
        lines.append(f"  {seq:13s} {role}  {100 * (x.w * x.score).sum() / x.w.sum():.1f}")
    second = chosen[((chosen.seq == "sequential_a") & (chosen.role == "B"))
                    | ((chosen.seq == "sequential_b") & (chosen.role == "A"))]
    share = 100 * (second.w * second.score).sum() / second.w.sum()
    lines.append(f"  second picker under a block sequence: {share:.1f}")
    return lines


def network_lines(recorded: pd.DataFrame) -> list[str]:
    f = recorded[recorded.side == "real"]
    weight = f.w.sum() / len(ALGS)
    chosen = chosen_rows(f, LEVELS["network and budget"])
    pagerank = f[f.player == "PR"]
    lines = ["Real networks, network and budget known: choice per budget (5, 10, 15, 20) and gain over "
             "always PageRank (percentage points of the real-network score)"]
    for network in s.REAL:
        x = chosen[chosen.graph_name == network]
        choice = x.drop_duplicates("k").sort_values("k").player.tolist()
        always = pagerank[pagerank.graph_name == network]
        gain = ((x.w * x.score).sum() - (always.w * always.score).sum()) / weight
        lines.append(f"  {network:14s} {'/'.join(choice):20s} {round(100 * gain, 2) + 0.0:5.2f}")
    return lines


def best_reply_lines(recorded: pd.DataFrame) -> list[str]:
    """Per real cell and opponent: is the algorithm best against all four opponents also the best
    reply to this one? Scores over the four sequences and two roles; ties to the first algorithm in
    alphabetical order, as in the levels."""
    f = recorded[recorded.side == "real"]
    against = f.groupby(CELL + ["opp", "player"]).score.mean().unstack("player").sort_index(axis=1)
    field = against.groupby(level=CELL).sum()
    field_best = field.idxmax(axis=1).reindex(against.index.droplevel("opp")).to_numpy()
    reply = against.idxmax(axis=1).to_numpy()
    is_best = against.eq(against.max(axis=1), axis=0).to_numpy()
    among = is_best[np.arange(len(against)), against.columns.get_indexer(field_best)]
    return [f"Real networks, {len(against)} cases (cell x opponent): the best against all four opponents is the "
            f"best reply in {100 * np.mean(reply == field_best):.1f}% (ties to the first in alphabetical order); "
            f"it is one of the best replies in {100 * among.mean():.1f}%; the best reply is unique in "
            f"{100 * (is_best.sum(axis=1) == 1).mean():.1f}%"]


def main() -> None:
    recorded, draw = fine_table(True), fine_table(False)
    table = levels(recorded, draw)
    table.to_csv(OUT / "levels.csv", index=False)
    lines = ["Share of games won, a draw counting half (%)", table.round(2).to_string(index=False), ""]
    lines += second_picker_lines(recorded) + [""] + network_lines(recorded) + [""] + best_reply_lines(recorded)
    text = "\n".join(lines)
    (OUT / "knowledge.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
