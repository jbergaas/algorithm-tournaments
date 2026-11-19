"""Games of the predictor analyses: the deterministic matches between two different structural
algorithms at k = 10 and 20 on the real networks and the first instance (i0) of each generated
setting, with the distance race of the seeds each player received and B's margin."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

NETWORKS = s.REAL + [f"{setting}-i0" for setting in s.SETTINGS]
BUDGETS = (10, 20)
KEY = ["graph_name", "k", "algorithm_a", "algorithm_b", "draft_order"]


def load() -> pd.DataFrame:
    """capture/out/matches.csv.gz joined with mechanisms/out/race_match.csv; margin_b = B's margin."""
    m = pd.read_csv(s.out_dir("capture") / "matches.csv.gz")
    m = m[m.k.isin(BUDGETS) & m.graph_name.isin(NETWORKS)]
    race = pd.read_csv(s.out_dir("mechanisms") / "race_match.csv").rename(columns={"allocation_order": "draft_order"})
    m = m.merge(race, on=KEY, how="left", validate="many_to_one")
    assert m.race_b.notna().all()
    return m.assign(margin_b=(m.spread_b - m.spread_a) / m.n)
