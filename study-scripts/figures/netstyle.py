"""Network colours and markers shared by the result figures, and the study constants and loaders
they take from tables/common.py.

The 11 real networks get the ten matplotlib tab10 colours and black, each with its own marker
shape. Identity is carried by colour and marker together: tab10 alone does not separate every pair
for colour-blind readers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tables"))

from common import (BUDGETS, ELO_SEED, NETWORKS, THETA_FRACTION, THETAS, cross_play_summaries,  # noqa: E402
                    read_graph, with_additional_adoption)

__all__ = ["BUDGETS", "ELO_SEED", "NETWORKS", "STYLE", "THETA_FRACTION", "THETAS", "cross_play_summaries", "read_graph",
           "with_additional_adoption"]

STYLE = {
    "Erdos02": ("#1f77b4", "o"), "ca-GrQc": ("#ff7f0e", "s"), "wiki-Vote": ("#2ca02c", "^"),
    "ca-HepPh": ("#d62728", "D"), "ca-CondMat": ("#9467bd", "v"), "email-Enron": ("#8c564b", "P"),
    "com-DBLP": ("#e377c2", "X"), "com-Youtube": ("#7f7f7f", "*"),
    "ca-HepTh": ("#bcbd22", "h"), "soc-Epinions1": ("#17becf", "<"), "ego-Facebook": ("#000000", ">"),
}
