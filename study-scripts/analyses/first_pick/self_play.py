"""Deterministic self-play: both players submit the same ranking, so only the picking sequence
separates them, and a fair sequence should leave a small lead |m_B|.

Per network and sequence (A-first block, strict alternation, snake): the median lead |m_B| in
percent of the nodes over the four structural algorithms, four budgets and 13 positive thresholds.
Writes out/self_play_by_graph.csv and out/self_play.txt (the Erdos02 example of the text).
Usage: python3 first_pick/self_play.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("first_pick")
SEQUENCES = ["sequential_a", "alternating_a", "snake"]


def main() -> None:
    d = s.self_play()
    d = d[(d.theta > 0) & d.algorithm_a.isin(s.STRUCTURAL) & d.allocation_order.isin(SEQUENCES)]
    lead = d.mean_margin_b.abs().groupby([d.graph_name, d.allocation_order])
    table = pd.DataFrame({"n": lead.size(), "lead_med": 100 * lead.median()}).reset_index()
    table = table.rename(columns={"allocation_order": "sequence"})
    table.to_csv(OUT / "self_play_by_graph.csv", index=False)
    erdos = table[table.graph_name == "Erdos02"].set_index("sequence").lead_med
    text = ("Erdos02, median self-play lead (% of the nodes): "
            + ", ".join(f"{seq} {erdos[seq]:.2f}" for seq in SEQUENCES))
    (OUT / "self_play.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
