"""Picking-sequence tables.

Reads the deterministic cross-play and self-play condition summaries at positive thresholds,
data/network_stats.csv and data/similarity.csv. A setting is a network, budget, threshold and
ordered pair of algorithms; its outcome is the sign of B's mean margin. Writes:
  data/outcome_changes.csv   per network, the settings whose outcome differs from A-first block
                             under B-first block, strict alternation and snake; swing_pp, the
                             median |change of B's margin| (pp) between B-first and A-first over
                             the settings where B-first changes the outcome; the mean top-20
                             overlap (shared_k20) and the degree CV; and how often the first block
                             wins under each block order
  data/selfplay_balance.csv  deterministic self-play: the median lead |m_B| (%) under A-first
                             block, strict alternation and snake, and the number of settings whose
                             lead is smaller, equal or larger than under A-first block
  data/role_advantage.csv    median |role advantage| (% of the network) per network and sequence
and prints the numbers quoted with them.
"""
import numpy as np
import pandas as pd

from common import NETWORKS, OUT_DATA, cross_play_summaries, output_path, self_play_summaries

SETTING = ["graph_name", "k", "theta", "algorithm_a", "algorithm_b"]


def outcome_changes(deterministic: pd.DataFrame, stats: pd.DataFrame, shared_k20: pd.Series) -> pd.DataFrame:
    margins = deterministic.pivot_table(index=SETTING, columns="allocation_order", values="mean_margin_b")
    signs = np.sign(margins)
    rows = []
    for network in NETWORKS:
        margin, sign = margins.loc[network], signs.loc[network]
        changed = {o: int((sign[o] != sign.sequential_a).sum()) for o in ("sequential_b", "alternating_a", "snake")}
        b_first_changes = sign.sequential_b != sign.sequential_a
        swing = (100 * (margin.sequential_b - margin.sequential_a)[b_first_changes]).abs().median()
        rows.append(dict(network=network, settings=len(margin), B_first=changed["sequential_b"],
                         altern=changed["alternating_a"], snake=changed["snake"], swing_pp=swing,
                         shared_k20=shared_k20[network], deg_cv=stats.loc[network, "deg_cv"],
                         first_block_wins_seq_a=int((sign.sequential_a < 0).sum()),
                         first_block_wins_seq_b=int((sign.sequential_b > 0).sum())))
    return pd.DataFrame(rows).set_index("network")


def selfplay_balance(self_play: pd.DataFrame) -> pd.DataFrame:
    deterministic = self_play[~self_play.random_involving & (self_play.theta > 0)]
    leads = deterministic.assign(lead=deterministic.mean_margin_b.abs()).pivot_table(
        index=["graph_name", "algorithm_a", "k", "theta"], columns="allocation_order", values="lead")
    rows = []
    for network in NETWORKS:
        lead = leads.loc[network]
        row = dict(network=network, settings=len(lead))
        for sequence in ("sequential_a", "alternating_a", "snake"):
            row[f"med_{sequence}"] = 100 * lead[sequence].median()
        for sequence in ("alternating_a", "snake"):
            difference = lead[sequence] - lead.sequential_a
            row[f"{sequence}_smaller"] = int((difference < 0).sum())
            row[f"{sequence}_equal"] = int((difference == 0).sum())
            row[f"{sequence}_larger"] = int((difference > 0).sum())
        row["snake_over_block"] = row["med_snake"] / row["med_sequential_a"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("network")


def role_advantage(deterministic: pd.DataFrame) -> pd.DataFrame:
    """Advantage of the A role for every unordered pair of algorithms in a setting and sequence.

    It is minus the mean of B's margins over the two role assignments, so it is positive when the
    player in role A is favoured.
    """
    keys = ["graph_name", "k", "theta", "allocation_order", "algorithm_a", "algorithm_b"]
    one_way = deterministic[deterministic.algorithm_a < deterministic.algorithm_b]
    swapped = deterministic.rename(columns={"algorithm_a": "algorithm_b", "algorithm_b": "algorithm_a",
                                            "mean_margin_b": "swapped_margin_b"})
    pairs = one_way.merge(swapped[keys + ["swapped_margin_b"]], on=keys, how="left", validate="one_to_one")
    assert pairs.swapped_margin_b.notna().all()
    return pairs[keys[:4]].assign(role_adv=-(pairs.mean_margin_b + pairs.swapped_margin_b) / 2)


def print_role_advantage(advantage: pd.DataFrame) -> None:
    def first_picker(sequence_rows: pd.DataFrame) -> pd.Series:
        # Under B-first block the first picker is B, whose advantage is minus A's.
        return pd.concat([sequence_rows[sequence_rows.allocation_order == "sequential_a"].role_adv,
                          -sequence_rows[sequence_rows.allocation_order == "sequential_b"].role_adv])

    block = first_picker(advantage)
    print(f"\nblock sequences: first picker favoured {100 * (block > 0).mean():.1f}%, median adv "
          f"{100 * block.median():.2f}%, median |adv| {100 * block.abs().median():.2f}%")
    for sequence in ("alternating_a", "snake"):
        values = advantage[advantage.allocation_order == sequence].role_adv
        print(f"  {sequence}: A favoured {100 * (values > 0).mean():.1f}%, "
              f"median |adv| {100 * values.abs().median():.2f}%")
    erdos = first_picker(advantage[advantage.graph_name == "Erdos02"])
    print(f"  Erdos02 block: median first-picker adv {100 * erdos.median():.2f}%")


def main() -> None:
    pd.set_option("display.width", 250)
    stats = pd.read_csv(f"{OUT_DATA}/network_stats.csv").set_index("network")
    similarity = pd.read_csv(f"{OUT_DATA}/similarity.csv")
    shared_k20 = similarity[similarity.k == 20].set_index("network")["mean"]

    summaries = cross_play_summaries()
    deterministic = summaries[~summaries.random_involving & ~summaries.is_self_play & (summaries.theta > 0)]

    changes = outcome_changes(deterministic, stats, shared_k20)
    changes.to_csv(output_path(OUT_DATA, "outcome_changes.csv"))
    print(f"\noutcome changes ({changes.settings.iloc[0]} settings per network):")
    print(changes.round(3).to_string())

    balance = selfplay_balance(self_play_summaries())
    balance.to_csv(output_path(OUT_DATA, "selfplay_balance.csv"))
    print("\nself-play balance:")
    print(balance.round(3).to_string())

    advantage = role_advantage(deterministic)
    print_role_advantage(advantage)
    median_advantage = advantage.assign(a=advantage.role_adv.abs() * 100).pivot_table(
        index="graph_name", columns="allocation_order", values="a", aggfunc="median").loc[NETWORKS]
    median_advantage.to_csv(output_path(OUT_DATA, "role_advantage.csv"))
    print("  median |role advantage| (% of network):")
    print(median_advantage.round(2).to_string())


if __name__ == "__main__":
    main()
