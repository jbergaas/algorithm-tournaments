"""How often the player ahead on a property wins, by threshold, under the block sequences (left)
and under strict alternation and snake (right).

All three properties are counted on the same games: decided games between two different
structural algorithms at k = 10 and 20, from theta = 1/8 up to each budget's collapse, on the real
networks and one instance of each generated setting. Games where either player is level on any of
the three are left out. Each property is measured on the seeds the player received: the distance
race (more nodes nearer to its seeds), first-round capture (more nodes its seeds alone would win
in the first round) and access (more still-neutral nodes next to its first-round captures in the
match).

Usage: python3 predictors.py <same_games.csv.gz> <out.pdf>
same_games.csv.gz has one row per game with its theta, sequence (allocation order) and a true/false
flag per property (race_hit, capture_hit, access_hit) that is true when the player ahead on it won.
Writes <out>.pdf, a PNG preview and the plotted values as <out>.csv.
"""
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Above 1/4 most networks have collapsed and few games are left, so the plot stops there.
LAST_THETA = 1 / 4
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
LINES = [("race_hit", "Distance race", "#0072B2", "o", "-"),
         ("capture_hit", "First-round capture", "#D55E00", "s", "--"),
         ("access_hit", "Access", "#009E73", "^", ":")]
PANELS = [(("sequential_a", "sequential_b"), "Block sequences"),
          (("alternating_a", "snake"), "Strict alternation and snake")]


def main(games_csv: str, out: str) -> None:
    out = Path(out)
    games = pd.read_csv(games_csv)
    games = games[games.theta <= LAST_THETA + 1e-9]
    rows = []
    for sequences, title in PANELS:
        for theta, g in games[games.sequence.isin(sequences)].groupby("theta"):
            rows.append(dict(panel=title, theta=theta, games=len(g),
                             **{key: 100 * g[key].mean() for key, *_ in LINES}))
    data = pd.DataFrame(rows)
    data.to_csv(out.with_suffix(".csv"), index=False)

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (_, title) in zip(axes, PANELS):
        d = data[data.panel == title]
        for key, label, colour, marker, style in LINES:
            ax.plot(d.theta, d[key], color=colour, marker=marker, linestyle=style, linewidth=1.6, markersize=5,
                    label=label)
        ax.axhline(50, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=9, color=INK)
        thetas = sorted(d.theta.unique())
        ax.set_xticks(thetas, [str(Fraction(t).limit_denominator(64)) for t in thetas], fontsize=8, color=MUTED)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    axes[1].text(LAST_THETA, 51, "no link", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    axes[0].set_ylabel("Player ahead wins (% of decided games)", fontsize=9, color=INK)
    axes[0].set_ylim(45, 100)
    fig.supxlabel(r"Threshold $\theta$", fontsize=9, color=INK, y=-0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=180)
    print(data.round(1).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
