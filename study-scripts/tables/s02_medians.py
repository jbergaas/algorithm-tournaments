"""Adoption and round medians, threshold regimes and the collapse of every network.

Reads the cross-play condition summaries and writes:
  data/medians.csv   graph_name, k, theta, allocation_order, q_pct, rounds: the median over the
                     20 ordered cross-algorithm condition means of q (additional acquisition, % of
                     all nodes) and of the acquisition rounds
  data/regimes.csv   graph_name, k, theta, median_adoption (seeds included), conditions, regime
  data/collapse.csv  per network: the largest drop of the k=20 snake median q between neighbouring
                     thresholds (drop_*), and the regime collapse at k=20, i.e. the last R2
                     threshold and the first R3 threshold above it (regime_*)
and prints the numbers quoted with them.
"""
import numpy as np
import pandas as pd

from common import (BUDGETS, NETWORKS, OUT_DATA, THETA_FRACTION, THETAS, cross_play_summaries, output_path,
                    threshold_regimes, with_additional_adoption)


def condition_medians(summaries: pd.DataFrame) -> pd.DataFrame:
    cross = summaries[~summaries.is_self_play]
    groups = cross.groupby(["graph_name", "k", "theta", "allocation_order"])
    medians = groups.agg(q_pct=("q", "median"), rounds=("mean_acquisition_rounds", "median"),
                         n=("q", "size")).reset_index()
    assert (medians.n == 20).all()
    medians["q_pct"] *= 100
    return medians.drop(columns="n")


def largest_drop(q_by_theta: np.ndarray) -> dict:
    """Neighbouring thresholds with the largest fall of the median q, and q at both ends."""
    i = int(np.argmax(q_by_theta[:-1] - q_by_theta[1:]))
    return {"drop_from": THETAS[i], "drop_to": THETAS[i + 1],
            "drop_q_from": q_by_theta[i], "drop_q_to": q_by_theta[i + 1]}


def regime_collapse(regime_by_theta: pd.Series) -> dict:
    """Last R2 threshold and the first R3 threshold above it (NaN if there is none).

    `contiguous` is False if an R2 threshold lies above an R3 threshold.
    """
    spreading = [t for t in THETAS if t > 1 / 16 and regime_by_theta[t] == "R2"]
    collapsed = [t for t in THETAS if t > 1 / 16 and regime_by_theta[t] == "R3"]
    last_spreading = max(spreading) if spreading else 1 / 16
    first_collapsed = min((t for t in collapsed if t > last_spreading), default=np.nan)
    contiguous = all(t < min(collapsed) for t in spreading) if collapsed and spreading else True
    return {"regime_last_R2": last_spreading, "regime_first_R3": first_collapsed, "regime_contiguous": contiguous,
            "regime_R2_thetas": " ".join(THETA_FRACTION[t] for t in spreading)}


def collapse_table(medians: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    snake20 = medians[(medians.allocation_order == "snake") & (medians.k == 20)]
    q_snake20 = snake20.pivot(index="graph_name", columns="theta", values="q_pct")
    rows = []
    for network in NETWORKS:
        at_k20 = regimes[(regimes.graph_name == network) & (regimes.k == 20)].set_index("theta")
        rows.append({"network": network,
                     **largest_drop(q_snake20.loc[network, THETAS].values),
                     **regime_collapse(at_k20.regime)})
    return pd.DataFrame(rows).set_index("network")


def print_spreading_thresholds(regimes: pd.DataFrame) -> None:
    print("\nR2 thresholds per network and k:")
    for network in NETWORKS:
        spreading = regimes[(regimes.graph_name == network) & (regimes.regime == "R2")]
        per_budget = (f"k={k}: " + ",".join(THETA_FRACTION[t] for t in sorted(spreading[spreading.k == k].theta))
                      for k in BUDGETS)
        print(f"  {network:13s} " + " | ".join(per_budget))


def print_sequences_near_snake(medians: pd.DataFrame) -> None:
    """Share of network-threshold pairs (k=20) where all three other sequences are within 1 pp of snake."""
    wide = medians[medians.k == 20].pivot_table(index=["graph_name", "theta"], columns="allocation_order",
                                                values="q_pct")
    wide = wide.loc[[(network, t) for network in NETWORKS for t in THETAS]]
    gap = wide[["sequential_a", "sequential_b", "alternating_a"]].sub(wide.snake, axis=0).abs()
    all_within = (gap <= 1).all(axis=1)
    print(f"k=20: all three sequences within 1 pp of snake in {all_within.sum()}/{len(wide)} = "
          f"{100 * all_within.mean():.1f}% of network-threshold combinations")


def main() -> None:
    summaries = with_additional_adoption(cross_play_summaries())
    medians = condition_medians(summaries)
    medians.to_csv(output_path(OUT_DATA, "medians.csv"), index=False)

    regimes = threshold_regimes(summaries)
    regimes.to_csv(output_path(OUT_DATA, "regimes.csv"), index=False)

    collapse = collapse_table(medians, regimes)
    collapse.to_csv(output_path(OUT_DATA, "collapse.csv"))
    pd.set_option("display.width", 250)
    print(collapse.to_string())

    print_spreading_thresholds(regimes)
    print_sequences_near_snake(medians)


if __name__ == "__main__":
    main()
