"""Average performance, competitive ordering and comparison with Random.

Reads the cross-play condition summaries, the condition uncertainty table and data/collapse.csv.
The first five files written are at k=20 under snake and positive thresholds. Writes:
  data/average.csv                mean own margin (%) per network and algorithm, over the four
                                  opponents in both roles and the positive thresholds
  data/positive.csv               number of those opponent-and-role means that are positive
  data/best_by_theta.csv          highest average per network and threshold, and whether the
                                  threshold lies before or after the largest drop of adoption
                                  (drop_from of data/collapse.csv, not always the collapse interval
                                  of the thesis)
  data/competitive_ordering.csv   algorithms with a positive mean margin against every other
                                  algorithm in both roles, per network and threshold ("--": none)
  data/competitive_min_leads.csv  the smallest of those margins, in nodes
  data/random_comparisons.csv     per network and structural algorithm, the settings (budgets,
                                  sequences, positive thresholds, both roles) with a positive mean
                                  margin against Random
  data/random_degree.csv          Random (A) against Degree (B) at k=10, theta=3/16, snake
and prints the numbers quoted with them.
"""
import pandas as pd

from common import (ALGORITHMS, LETTER, NETWORKS, NODES, OUT_DATA, POSITIVE_THETAS, STRUCTURAL, THETA_FRACTION,
                    condition_uncertainty, cross_play_summaries, output_path, own_margins)

OPPONENT_ROLE_MEANS = 8   # four opponents, two roles


def count_positive(values: pd.Series) -> int:
    return int((values > 0).sum())


def average_performance(snake20: pd.DataFrame):
    """Per (network, theta, algorithm): mean own margin, number of means and how many are positive."""
    per_threshold = own_margins(snake20).groupby(["graph_name", "theta", "alg"]).own.agg(
        ["mean", "size", count_positive])
    per_threshold.columns = ["mean", "n", "beats"]
    assert (per_threshold.n == OPPONENT_ROLE_MEANS).all()
    mean_margin = (100 * per_threshold["mean"]).unstack("alg").groupby(level="graph_name").mean()
    positive = per_threshold["beats"].unstack("alg").groupby(level="graph_name").sum()
    return per_threshold, mean_margin.loc[NETWORKS][ALGORITHMS], positive.loc[NETWORKS][ALGORITHMS]


def print_average(mean_margin: pd.DataFrame, positive: pd.DataFrame) -> None:
    print(f"\nTable tab:results-average: mean own margin (%) over {len(POSITIVE_THETAS)} positive thresholds | "
          f"positive means (of {OPPONENT_ROLE_MEANS * len(POSITIVE_THETAS)})")
    print(pd.concat([mean_margin.rename(columns=LETTER).round(1), positive.rename(columns=LETTER)], axis=1).to_string())
    print("  best average per network:", mean_margin[STRUCTURAL].idxmax(axis=1).map(LETTER).to_dict())
    most_positive = {network: "".join(LETTER[a] for a in STRUCTURAL
                                      if positive.loc[network, a] == positive.loc[network, STRUCTURAL].max())
                     for network in NETWORKS}
    print("  most positive means per network:", most_positive)
    print("  PageRank average positive on all:", bool((mean_margin.pagerank > 0).all()),
          "; Random negative on all:", bool((mean_margin.random < 0).all()))
    print("  Degree negative average on:", list(mean_margin.index[mean_margin.degree < 0]))
    print("  Degree fewest positive means among structural on:",
          [n for n in NETWORKS if positive.loc[n, "degree"] == positive.loc[n, STRUCTURAL].min()])


def best_by_threshold(per_threshold: pd.DataFrame, last_before_drop: pd.Series) -> pd.DataFrame:
    """Highest average per network and threshold; phase "before" up to the lower end of the largest
    drop of the k=20 snake median adoption (inclusive), "after" above it."""
    rows = []
    for (network, theta), means in per_threshold["mean"].groupby(level=["graph_name", "theta"]):
        by_algorithm = means.droplevel([0, 1])
        last_before = last_before_drop[network]
        rows.append(dict(network=network, theta=theta, best=LETTER[by_algorithm.idxmax()],
                         best_mean_pct=100 * by_algorithm.max(), phase="before" if theta <= last_before else "after"))
    best = pd.DataFrame(rows)

    before, after = best[best.phase == "before"], best[best.phase == "after"]
    print(f"  best average before the collapse (incl. last threshold before it): "
          f"{before.best_mean_pct.min():.1f}..{before.best_mean_pct.max():.1f}% over {len(before)} network-thresholds; "
          f"after: {after.best_mean_pct.min():.1f}..{after.best_mean_pct.max():.1f}%, "
          f"PageRank best in {int((after.best == 'P').sum())} of {len(after)}; "
          f"counts {after.best.value_counts().to_dict()}")
    columns = ["network", "theta", "best", "best_mean_pct"]
    print("  before-collapse best values below 25%:",
          before[before.best_mean_pct < 25][columns].round(1).values.tolist())
    print("  after-collapse best values >= 8%:", after[after.best_mean_pct >= 8][columns].round(1).values.tolist())
    return best


def competitive_ordering(snake20: pd.DataFrame):
    """Algorithms whose mean margin is positive against every opponent in both roles.

    Returns {(network, theta): letters or "--"} and {(network, theta): smallest such margin in
    nodes}; if several algorithms qualify, the last one's margin is kept.
    """
    qualifiers, min_leads = {}, {}
    for (network, theta), group in snake20.groupby(["graph_name", "theta"]):
        letters = []
        for algorithm in ALGORITHMS:
            own = pd.concat([-group[group.algorithm_a == algorithm].mean_margin_b,
                             group[group.algorithm_b == algorithm].mean_margin_b])
            if (own > 0).all():
                letters.append(LETTER[algorithm])
                min_leads[(network, theta)] = own.min() * NODES[network]
        qualifiers[(network, theta)] = "".join(letters) or "--"
    return qualifiers, min_leads


def print_competitive(ordering: pd.DataFrame, qualifiers: dict, min_leads: dict, per_threshold: pd.DataFrame) -> None:
    print("  Table tab:results-competitive-ordering:")
    print(ordering.to_string())
    print("  thresholds with no qualifier per network:", (ordering == "--").sum(axis=1).to_dict(),
          f"(of {len(POSITIVE_THETAS)})")
    letters = set("".join(ordering.values.ravel()).replace("-", ""))
    everywhere = {n: [a for a in "DCPB" if all(a in ordering.loc[n, c] for c in ordering.columns)] for n in NETWORKS}
    print("  algorithms that qualify somewhere:", sorted(letters),
          "; any algorithm qualifying at every threshold on a network:", everywhere)
    agree = sum(1 for (n, t), q in qualifiers.items()
                if q != "--" and LETTER[per_threshold.loc[(n, t), "mean"].idxmax()] == q)
    print(f"  qualifier = highest average at that threshold: {agree} of "
          f"{sum(1 for q in qualifiers.values() if q != '--')}")
    print("  min leads at 3/4 (nodes):",
          {n: round(min_leads[(n, 0.75)], 2) for n in NETWORKS if (n, 0.75) in min_leads})


def random_comparisons(cross: pd.DataFrame) -> pd.DataFrame:
    positive = cross[cross.theta.isin(POSITIVE_THETAS)]
    with_random = positive[(positive.algorithm_a == "random") ^ (positive.algorithm_b == "random")]
    own = own_margins(with_random)
    own = own[own.alg != "random"]
    own = own.assign(own_nodes=own.own * own.graph_name.map(NODES))
    counts = own.groupby(["graph_name", "alg"]).agg(pos=("own", count_positive), n=("own", "size"))
    table = counts["pos"].unstack("alg").loc[NETWORKS][STRUCTURAL]

    settings = counts.n.iloc[0]
    print(f"  Table tab:results-random-comparisons, settings per cell {settings}:")
    print(table.to_string())
    exceptions = {f"{n}/{a}": int(table.loc[n, a]) for n in NETWORKS for a in STRUCTURAL if table.loc[n, a] < settings}
    print(f"  combinations with all settings positive: {int((table == settings).sum().sum())} of {table.size}; "
          "exceptions:", exceptions)
    not_positive = own[own.own <= 0]
    below_one_node = not_positive[not_positive.own_nodes.abs() < 1]
    print(f"  non-positive cases {len(not_positive)}; with |mean node difference| < 1: {len(below_one_node)} "
          f"at thetas {sorted(THETA_FRACTION[t] for t in below_one_node.theta.unique())}; "
          f"zero means {int((not_positive.own == 0).sum())}")
    print("  thresholds of the exceptions:", not_positive.groupby(["graph_name", "alg"]).theta.apply(
        lambda v: sorted(THETA_FRACTION[t] for t in v.unique())).to_dict())
    return table


def random_against_degree(cross: pd.DataFrame, uncertainty: pd.DataFrame) -> pd.DataFrame:
    def one_setting(table: pd.DataFrame) -> pd.DataFrame:
        return table[(table.k == 10) & (table.theta == 0.1875) & (table.allocation_order == "snake")
                     & (table.algorithm_a == "random") & (table.algorithm_b == "degree")].set_index("graph_name")

    summary, interval = one_setting(cross), one_setting(uncertainty)
    nodes = summary.index.map(NODES)
    return pd.DataFrame({"mean_A_owned": summary.mean_a_fraction * nodes,
                         "mean_B_owned": summary.mean_b_fraction * nodes,
                         "mean_margin_b": summary.mean_margin_b,
                         "lower": interval.interval_lower_95, "upper": interval.interval_upper_95,
                         "degree_wins": summary.strict_b_win_proportion * 100,
                         "draws": summary.draw_proportion * 100}).loc[NETWORKS]


def main() -> None:
    pd.set_option("display.width", 250)
    summaries = cross_play_summaries()
    cross = summaries[~summaries.is_self_play]
    last_before_drop = pd.read_csv(f"{OUT_DATA}/collapse.csv", index_col=0).drop_from
    snake20 = cross[(cross.k == 20) & (cross.allocation_order == "snake") & cross.theta.isin(POSITIVE_THETAS)]

    per_threshold, mean_margin, positive = average_performance(snake20)
    mean_margin.rename(columns=LETTER).to_csv(output_path(OUT_DATA, "average.csv"))
    positive.rename(columns=LETTER).to_csv(output_path(OUT_DATA, "positive.csv"))
    print_average(mean_margin, positive)

    best_by_threshold(per_threshold, last_before_drop).to_csv(output_path(OUT_DATA, "best_by_theta.csv"), index=False)

    qualifiers, min_leads = competitive_ordering(snake20)
    ordering = pd.DataFrame({THETA_FRACTION[t]: [qualifiers[(n, t)] for n in NETWORKS] for t in POSITIVE_THETAS},
                            index=NETWORKS)
    ordering.to_csv(output_path(OUT_DATA, "competitive_ordering.csv"))
    pd.Series(min_leads, name="min_lead_nodes").rename_axis(["network", "theta"]).to_csv(
        output_path(OUT_DATA, "competitive_min_leads.csv"))
    print_competitive(ordering, qualifiers, min_leads, per_threshold)

    random_comparisons(cross).to_csv(output_path(OUT_DATA, "random_comparisons.csv"))

    random_degree = random_against_degree(cross, condition_uncertainty())
    random_degree.to_csv(output_path(OUT_DATA, "random_degree.csv"))
    print("\nRandom (A) vs Degree (B), k=10, theta=3/16, snake:")
    print(random_degree.round(4).to_string())
    k5 = cross[(cross.graph_name == "ca-HepPh") & (cross.k == 5) & (cross.theta == 0.1875)
               & (cross.allocation_order == "snake") & (cross.algorithm_a == "random")
               & (cross.algorithm_b == "degree")].iloc[0]
    print(f"ca-HepPh k=5: Degree wins {k5.strict_b_win_proportion*100:.0f} of 100, mean margin {k5.mean_margin_b:.3f}")


if __name__ == "__main__":
    main()
