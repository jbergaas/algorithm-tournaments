"""Before or after the collapse, and the first-pick advantage, as the first-pick analysis uses them.

Only the four structural algorithms take part, so the median adoption that places the collapse
runs over their cross-algorithm conditions (12 ordered pairs, four sequences) of a network, budget
and positive threshold; the spread is "before the collapse" when that median is at least one half.

The first-pick advantage of a pair of different structural algorithms in one setting (network,
budget, threshold, sequence) is the own margin of whoever picks first, averaged over the two role
assignments.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

CELL = ["graph_name", "k", "theta"]


def family_of(network: str) -> str:
    return "real" if network in s.REAL else network.split("-")[0]


def deterministic(d: pd.DataFrame) -> pd.DataFrame:
    """Conditions between two different structural algorithms at a positive threshold."""
    return d[(d.theta > 0) & ~d.random_involving & (d.algorithm_a != d.algorithm_b)]


def spreading(d: pd.DataFrame) -> pd.DataFrame:
    """Median adoption of the structural conditions per network, budget and threshold, and whether
    it is at least one half."""
    det = deterministic(d)
    median = (1 - det.mean_neutral_fraction).groupby([det.graph_name, det.k, det.theta]).median()
    out = median.rename("median_adoption").reset_index()
    out["spread"] = out.median_adoption >= 0.5
    return out


def role_advantage(d: pd.DataFrame) -> pd.DataFrame:
    """One row per network, budget, threshold, sequence and unordered pair: m_fwd and m_rev are B's
    margins in the two role assignments, fp_adv the first picker's mean margin, and how the two
    matches ended for the first picker (fp_both: it wins both; sp_both: it loses both; alg_both: the
    same algorithm wins both)."""
    key = CELL + ["allocation_order"]
    m = deterministic(d)[key + ["algorithm_a", "algorithm_b", "mean_margin_b"]]
    rev = m.rename(columns={"algorithm_a": "algorithm_b", "algorithm_b": "algorithm_a", "mean_margin_b": "m_rev"})
    j = m.merge(rev, on=key + ["algorithm_a", "algorithm_b"])
    j = j[j.algorithm_a < j.algorithm_b].rename(columns={"mean_margin_b": "m_fwd"})
    first_is_a = j.allocation_order != "sequential_b"
    adv = -(j.m_fwd + j.m_rev) / 2
    j["fp_adv"] = np.where(first_is_a, adv, -adv)
    fwd = np.where(first_is_a, -j.m_fwd, j.m_fwd)
    rev_fp = np.where(first_is_a, -j.m_rev, j.m_rev)
    j["fp_both"] = (fwd > 0) & (rev_fp > 0)
    j["sp_both"] = (fwd < 0) & (rev_fp < 0)
    j["alg_both"] = ((fwd > 0) & (rev_fp < 0)) | ((fwd < 0) & (rev_fp > 0))
    return j.merge(spreading(d), on=CELL)
