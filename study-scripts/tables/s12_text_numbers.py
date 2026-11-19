"""Numbers quoted in the text of the results that s02-s08 do not print.

Reads the CSVs written by s01-s07, the cross-play condition summaries and rankings.csv of the
wiki-Vote batches at theta = 1/16 in runs/. It only prints. Keep the output with
`python3 s12_text_numbers.py | tee out/data/text_numbers.out`.
"""
import csv

import numpy as np
import pandas as pd

from common import (ALGORITHMS, BUDGETS, ELO_SEED, LETTER, NETWORKS, OUT_DATA, SEQUENCES, STRUCTURAL,
                    THETA_FRACTION, batch_dir, cross_play_summaries, is_deterministic_cross_play)

csv.field_size_limit(10**9)
pd.set_option("display.width", 250)


def print_snake_tables(q20: pd.DataFrame, rounds20: pd.DataFrame) -> None:
    print("== k=20 snake median q (%), 14 thresholds")
    print(q20.rename(columns=THETA_FRACTION).round(1).to_string())
    print("\n== k=20 snake median rounds, 14 thresholds")
    print(rounds20.rename(columns=THETA_FRACTION).round(1).to_string())


def print_collapse(stats: pd.DataFrame, collapse: pd.DataFrame) -> None:
    print("\n== collapse (largest drop of the k=20 snake median) vs average degree")
    for network in stats.loc[NETWORKS].avg_deg.sort_values(ascending=False).index:
        c = collapse.loc[network]
        print(f"  {network:13s} avg deg {stats.loc[network, 'avg_deg']:5.1f}  largest drop "
              f"{THETA_FRACTION[c.drop_from]}->{THETA_FRACTION[c.drop_to]}  regime collapse (k=20) "
              f"{THETA_FRACTION[c.regime_last_R2]}->{THETA_FRACTION[c.regime_first_R3]}")


def print_rounds(stats: pd.DataFrame, collapse: pd.DataFrame, q20: pd.DataFrame, rounds20: pd.DataFrame) -> None:
    print("\n== rounds: theta=0 vs distance; peak vs last threshold before the largest drop")
    for network in NETWORKS:
        peak = rounds20.loc[network].idxmax()
        last_before = collapse.loc[network, "drop_from"]
        print(f"  {network:13s} distance {stats.loc[network, 'dist_from_top40']:2d} rounds@0 "
              f"{rounds20.loc[network, 0.0]:5.2f} | peak {rounds20.loc[network].max():5.1f} at "
              f"{THETA_FRACTION[peak]:5s} (q there {q20.loc[network, peak]:5.1f}%) | last before drop "
              f"{THETA_FRACTION[last_before]} | peak==last-before-drop: {peak == last_before}")
    print(f"  com-DBLP at 1/4: q {q20.loc['com-DBLP', 0.25]:.2f}%, rounds {rounds20.loc['com-DBLP', 0.25]:.1f}")


def print_adoption(stats: pd.DataFrame, q20: pd.DataFrame, snake: pd.DataFrame) -> None:
    print("\n== q at 5/16 and 3/4 (k=20 snake), sorted")
    for theta in (0.3125, 0.75):
        print(f"  {THETA_FRACTION[theta]}:", q20[theta].sort_values(ascending=False).round(1).to_dict())
    print("  q(1/8) vs largest component:",
          {n: (round(q20.loc[n, 0.125], 1), round(100 * stats.loc[n, 'largest'], 1)) for n in NETWORKS})

    print("\n== budget tipping points (snake median q, %, k = 5, 10, 15, 20)")
    for network, theta in (("ca-CondMat", 0.25), ("com-Youtube", 0.25), ("ca-HepTh", 0.25), ("soc-Epinions1", 0.15625),
                           ("ca-HepPh", 0.21875), ("Erdos02", 0.3125), ("email-Enron", 0.3125)):
        setting = snake[(snake.graph_name == network) & (snake.theta == theta)]
        by_budget = setting.sort_values("k").q_pct.round(1).tolist()
        print(f"  {network:13s} theta {THETA_FRACTION[theta]:5s}: {by_budget}")


def print_erdos_block_winners(summaries: pd.DataFrame) -> None:
    deterministic = summaries[~summaries.random_involving & (summaries.theta > 0)]
    margins = deterministic.pivot_table(index=["graph_name", "k", "theta", "algorithm_a", "algorithm_b"],
                                        columns="allocation_order", values="mean_margin_b")
    print("\n== Erdos02: settings where the first block does not win")
    erdos = margins.loc["Erdos02"]
    print((100 * erdos[(erdos.sequential_a >= 0) | (erdos.sequential_b <= 0)]).round(3).to_string())
    reversed_outcome = erdos[np.sign(erdos.sequential_a) != np.sign(erdos.sequential_b)]
    print(f"  B-first changes the outcome in {len(reversed_outcome)} of {len(erdos)}; first block wins under A-first "
          f"{int((erdos.sequential_a < 0).sum())}, under B-first {int((erdos.sequential_b > 0).sum())}")


def print_wiki_vote_seed_sets() -> None:
    print("\n== wiki-Vote: do snake and A-first block give different seed sets? (deterministic pairs, by k)")
    for k in BUDGETS:
        seeds = {}
        with open(f"{batch_dir('wiki-Vote', k, 0.0625)}/rankings.csv") as f:
            for row in csv.DictReader(f):
                if is_deterministic_cross_play(row):
                    key = (row["algorithm_a"], row["algorithm_b"], row["draft_order"])
                    seeds[key] = (row["seeds_a"], row["seeds_b"])
        pairs = {(a, b) for a, b, _ in seeds}
        different = sum(seeds[(a, b, "snake")] != seeds[(a, b, "sequential_a")] for a, b in pairs)
        print(f"  k={k}: {different} of {len(pairs)} ordered pairs (seeds do not depend on theta)")


def print_common_setting(summaries: pd.DataFrame) -> None:
    print("\n== allocation common setting (Coverage Greedy A, PageRank B, k=20, 1/8): PageRank wins under")
    setting = summaries[(summaries.k == 20) & (summaries.theta == 0.125) & (summaries.algorithm_a == "coverage_greedy")
                        & (summaries.algorithm_b == "pagerank")]
    margins = setting.pivot(index="graph_name", columns="allocation_order", values="mean_margin_b")
    margins = margins.loc[NETWORKS, SEQUENCES]
    for network in NETWORKS:
        print(f"  {network:13s} " + ", ".join(f"{s} {100 * margins.loc[network, s]:+.2f}" for s in SEQUENCES))


def print_best_by_threshold() -> None:
    print("\n== best average per threshold (k=20 snake), PageRank as best after the largest drop")
    best = pd.read_csv(f"{OUT_DATA}/best_by_theta.csv")
    after = best[(best.phase == "after") & (best.best == "P")].sort_values("best_mean_pct", ascending=False)
    largest = [(r.network, THETA_FRACTION[round(r.theta, 6)], round(r.best_mean_pct, 1))
               for r in after.head(5).itertuples()]
    print(f"  PageRank best in {len(after)} of {int((best.phase == 'after').sum())} pairs after the drop; "
          "largest leads:", largest)
    before = best[best.phase == "before"]
    print(f"  before the drop: best average {before.best_mean_pct.min():.1f}..{before.best_mean_pct.max():.1f}% "
          f"(min at {before.loc[before.best_mean_pct.idxmin(), ['network', 'theta']].tolist()}, max at "
          f"{before.loc[before.best_mean_pct.idxmax(), ['network', 'theta']].tolist()})")
    print("  wiki-Vote best by threshold:",
          {THETA_FRACTION[round(r.theta, 6)]: r.best for r in best[best.network == "wiki-Vote"].itertuples()})


def print_elo_summaries() -> None:
    print(f"\n== Elo summaries (seed {ELO_SEED})")
    ratings = pd.read_csv(f"{OUT_DATA}/elo_all.csv")
    ratings = ratings[ratings.seed == ELO_SEED]
    for (cutoff, regime), group in ratings.groupby(["cutoff", "regime"]):
        elo = group.pivot(index="network", columns="alg", values="elo").loc[NETWORKS]
        firsts = elo[STRUCTURAL + ["random"]].idxmax(axis=1).map(LETTER).value_counts().to_dict()
        lowest = elo[STRUCTURAL].idxmin(axis=1).map(LETTER).value_counts().to_dict()
        random_last = bool((elo.idxmin(axis=1) == "random").all())
        print(f"  [{cutoff:7s} {regime:7s}] means " + ", ".join(f"{LETTER[a]} {elo[a].mean():.0f}" for a in ALGORITHMS)
              + f" | first on {firsts} | lowest structural {lowest} | Random last everywhere: {random_last}")
    overall = ratings[(ratings.cutoff == "std") & (ratings.regime == "ALL")]
    structural = overall.pivot(index="network", columns="alg", values="elo").loc[NETWORKS][STRUCTURAL]
    print("  overall: spread of the four structural ratings:",
          (structural.max(axis=1) - structural.min(axis=1)).round(0).astype(int).to_dict())
    print("  overall: Degree rank among the structural:",
          structural.rank(axis=1, ascending=False).degree.astype(int).to_dict())
    average = pd.read_csv(f"{OUT_DATA}/average.csv", index_col=0)
    print("  highest average margin:", average[list("DCPB")].idxmax(axis=1).to_dict())


def main() -> None:
    stats = pd.read_csv(f"{OUT_DATA}/network_stats.csv").set_index("network")
    medians = pd.read_csv(f"{OUT_DATA}/medians.csv")
    medians["theta"] = medians.theta.round(6)
    snake = medians[medians.allocation_order == "snake"]
    q20 = snake[snake.k == 20].pivot(index="graph_name", columns="theta", values="q_pct").loc[NETWORKS]
    rounds20 = snake[snake.k == 20].pivot(index="graph_name", columns="theta", values="rounds").loc[NETWORKS]
    collapse = pd.read_csv(f"{OUT_DATA}/collapse.csv", index_col=0)

    print_snake_tables(q20, rounds20)
    print_collapse(stats, collapse)
    print_rounds(stats, collapse, q20, rounds20)
    print_adoption(stats, q20, snake)

    summaries = cross_play_summaries()
    print_erdos_block_winners(summaries)
    print_wiki_vote_seed_sets()
    print_common_setting(summaries)
    print_best_by_threshold()
    print_elo_summaries()


if __name__ == "__main__":
    main()
