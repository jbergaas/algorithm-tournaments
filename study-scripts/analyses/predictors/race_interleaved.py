"""How often the leader of the distance race wins under strict alternation and snake, at the two
lowest thresholds.

Games (predictors/games.py): deterministic matches between two different structural algorithms
at k = 10 and 20 on the real networks and the first instance of each generated setting, under
strict alternation or snake, decided (|margin| > 1% of the nodes) and with a race leader
(race_b != 0). A hit is a game won by the race leader.

Writes out/race_interleaved.txt.
Usage: python3 predictors/race_interleaved.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import games  # noqa: E402
import shared as s  # noqa: E402

OUT = s.out_dir("predictors")


def main() -> None:
    m = games.load()
    m = m[m.draft_order.isin(s.INTERLEAVED) & (m.margin_b.abs() > 0.01) & (m.race_b != 0)]
    lines = [f"Race leader wins, strict alternation and snake, k = 10 and 20, {len(games.NETWORKS)} networks:"]
    for theta in (0.0, 1 / 16):
        x = m[m.theta == theta]
        hits = np.sign(x.margin_b) == np.sign(x.race_b)
        lines.append(f"  theta {s.THETA_LABEL[theta]:>4}: {100 * hits.mean():.1f}% of {len(x)} decided games "
                     f"on {x.graph_name.nunique()} networks")
    text = "\n".join(lines) + "\n"
    (OUT / "race_interleaved.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
