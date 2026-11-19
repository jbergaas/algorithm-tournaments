"""Threshold regimes and the games between the four structural algorithms, as the ratings use them.

Regimes per network, budget and threshold: low (theta <= 1/16), spread (theta > 1/16 and median
adoption at least one half) and above (theta > 1/16, median below one half). The median adoption
runs over all 80 cross-algorithm conditions (Random included) of that network, budget and threshold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

CELL = ["graph_name", "k", "theta"]


def regimes(cross_play: pd.DataFrame) -> pd.DataFrame:
    """Median adoption and regime label of every network, budget and threshold."""
    adoption = 1 - cross_play.mean_neutral_fraction
    median = adoption.groupby([cross_play.graph_name, cross_play.k, cross_play.theta]).median()
    out = median.rename("median_adoption").reset_index()
    out["regime"] = np.where(out.theta <= 1 / 16, "low", np.where(out.median_adoption >= 0.5, "spread", "above"))
    return out


def structural_games() -> pd.DataFrame:
    """Every condition between two different structural algorithms, in file order, with its regime."""
    d = s.cross_play()
    games = d[~d.random_involving].merge(regimes(d)[CELL + ["regime"]], on=CELL, how="left")
    assert games.algorithm_a.isin(s.STRUCTURAL).all() and games.algorithm_b.isin(s.STRUCTURAL).all()
    return games


def unit_of(network: str) -> str:
    """Independent unit: a real network, or a generated setting with its three instances."""
    return network if network in s.REAL else s.setting_of(network)
