"""Elo ratings of the algorithms on every network, with Bradley-Terry ratings as a check.

Every cross-algorithm condition (all budgets, sequences and both role assignments, Random
included) is one game between algorithm A and algorithm B. Elo uses K = 16, starts every algorithm
at 1500 and is averaged over 200 random game orders. The orders of one set of ratings come from one
numpy default_rng(seed), consumed network by network in table row order. Bradley-Terry ratings are
fitted with the MM algorithm (2000 iterations) and put on the Elo scale.

Games are rated per network in these sets (regime column; regimes from data/regimes.csv):
  ALL      every positive threshold
  R1       theta <= 1/16, theta = 0 included
  R2       theta > 1/16 and median adoption >= 1/2
  R3       theta > 1/16 and median adoption < 1/2
  R1_1/16  theta = 1/16 only (R1 without theta = 0, a check)
with three ways to score B (cutoff column):
  std      B's win proportion plus half its draw proportion
  1pc      1 if B's mean margin exceeds 1% of the nodes, 0 if A's does, 0.5 otherwise
  1pcruns  as 1pc, but a Random condition scores the share of its 100 runs that B wins by more
           than 1% plus half the share within +-1% (a check; per-run margins from s07a_random_runs)
std and 1pc are rated with the seeds 20261104 (ELO_SEED), 1 and 2, 1pcruns with 20261104.

Writes data/elo_all.csv (scope_name, cutoff, regime, network, alg, elo, bt, games, seed) and prints
the ratings and their spread over the seeds. scope_name is "ext" (the 11 networks and 14
thresholds) on every row. Analyses that read elo_all.csv select their rows by it.
"""
import numpy as np
import pandas as pd

from common import (ALGORITHMS, CONDITION_KEY, ELO_SEED, LETTER, NETWORKS, OUT_DATA, cross_play_summaries,
                    output_path)
from s07a_random_runs import load_random_runs, run_scores

SEEDS = (ELO_SEED, 1, 2)
REGIMES = ("ALL", "R1", "R2", "R3", "R1_1/16")
PLAYER_INDEX = {a: i for i, a in enumerate(ALGORITHMS)}


def bradley_terry(a: np.ndarray, b: np.ndarray, score: np.ndarray, n_players: int = 5) -> np.ndarray:
    """Bradley-Terry strengths by the MM algorithm (2000 iterations), on the Elo scale."""
    wins = np.zeros((n_players, n_players))
    games = np.zeros((n_players, n_players))
    np.add.at(wins, (b, a), score)
    np.add.at(wins, (a, b), 1 - score)
    np.add.at(games, (a, b), 1)
    np.add.at(games, (b, a), 1)
    strength = np.ones(n_players)
    for _ in range(2000):
        total_wins = wins.sum(axis=1)
        denominator = np.array([sum(games[i, j] / (strength[i] + strength[j]) for j in range(n_players) if j != i)
                                for i in range(n_players)])
        strength = np.maximum(total_wins, 1e-9) / denominator
        strength /= np.exp(np.mean(np.log(strength)))
    return 400 * np.log10(strength) + 1500


def elo(a: np.ndarray, b: np.ndarray, score: np.ndarray, rng: np.random.Generator, orders: int = 200,
        k_factor: float = 16, n_players: int = 5) -> np.ndarray:
    """Mean Elo rating over `orders` random game orders, all orders updated side by side.

    The permutations are drawn one after the other from rng.
    """
    if len(a) == 0:
        return np.full(n_players, np.nan)
    permutations = np.stack([rng.permutation(len(a)) for _ in range(orders)])
    ratings = np.full((orders, n_players), 1500.0)
    order_rows = np.arange(orders)
    for step in range(len(a)):
        game = permutations[:, step]
        player_a, player_b = a[game], b[game]
        expected_b = 1 / (1 + 10 ** ((ratings[order_rows, player_a] - ratings[order_rows, player_b]) / 400))
        delta = k_factor * (score[game] - expected_b)
        ratings[order_rows, player_b] += delta
        ratings[order_rows, player_a] -= delta
    # Summed one order at a time; ratings.mean(axis=0) can differ in the last bits.
    total = np.zeros(n_players)
    for order in range(orders):
        total += ratings[order]
    return total / orders


def all_games() -> pd.DataFrame:
    """Every cross-play condition with the three scores of B and the regime label."""
    summaries = cross_play_summaries()
    games = summaries[~summaries.is_self_play].copy()
    games["score_std"] = games.strict_b_win_proportion + 0.5 * games.draw_proportion
    games["score_1pc"] = np.select([games.mean_margin_b > 0.01, games.mean_margin_b < -0.01], [1.0, 0.0], 0.5)
    per_run = run_scores(load_random_runs(), 0.01).rename(columns={"score_b": "score_runs"})
    games = games.merge(per_run, on=CONDITION_KEY, how="left", validate="one_to_one")
    assert games.score_runs.notna().sum() == games.random_involving.sum()
    games["score_1pcruns"] = games.score_1pc
    games.loc[games.random_involving, "score_1pcruns"] = games.loc[games.random_involving, "score_runs"]
    regimes = pd.read_csv(f"{OUT_DATA}/regimes.csv")
    regimes["theta"] = regimes.theta.round(6)
    games = games.merge(regimes[["graph_name", "k", "theta", "regime"]], on=["graph_name", "k", "theta"],
                        how="left", validate="many_to_one")
    assert games.regime.notna().all()
    return games


def games_in(games: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == "ALL":
        return games[games.theta > 0]
    if regime == "R1_1/16":
        return games[games.theta == 0.0625]
    return games[games.regime == regime]


def rate(games: pd.DataFrame, regime: str, cutoff: str, seed: int) -> list:
    rng = np.random.default_rng(seed)
    selected = games_in(games, regime)
    rows = []
    for network in NETWORKS:
        played = selected[selected.graph_name == network]
        a = played.algorithm_a.map(PLAYER_INDEX).to_numpy()
        b = played.algorithm_b.map(PLAYER_INDEX).to_numpy()
        score = played[f"score_{cutoff}"].to_numpy(dtype=float)
        elo_ratings = elo(a, b, score, rng)
        bt_ratings = bradley_terry(a, b, score) if len(a) else np.full(len(ALGORITHMS), np.nan)
        for i, algorithm in enumerate(ALGORITHMS):
            rows.append(dict(scope_name="ext", cutoff=cutoff, regime=regime, network=network, alg=algorithm,
                             elo=elo_ratings[i], bt=bt_ratings[i], games=len(played), seed=seed))
    return rows


def print_ratings(ratings: pd.DataFrame) -> None:
    main_seed = ratings[ratings.seed == ELO_SEED]
    for (cutoff, regime), group in main_seed.groupby(["cutoff", "regime"], sort=False):
        elo_table = group.pivot(index="network", columns="alg", values="elo").loc[NETWORKS, ALGORITHMS]
        bt_table = group.pivot(index="network", columns="alg", values="bt").loc[NETWORKS, ALGORITHMS]
        same_order = {n: list(elo_table.loc[n].sort_values(ascending=False).index)
                      == list(bt_table.loc[n].sort_values(ascending=False).index) for n in NETWORKS}
        same_winner = {n: elo_table.loc[n].idxmax() == bt_table.loc[n].idxmax() for n in NETWORKS}
        print(f"\n[{cutoff} {regime}] games per network: "
              f"{group.drop_duplicates('network').set_index('network').games.to_dict()}")
        print(elo_table.round(0).rename(columns=LETTER).to_string())
        print("  BT order equals Elo order:", [n for n, same in same_order.items() if not same] or "all",
              "| BT winner differs on:", [n for n, same in same_winner.items() if not same] or "none")
    spread = ratings.groupby(["cutoff", "regime", "network", "alg"]).elo.agg(lambda v: v.max() - v.min())
    print(f"\nElo spread over seeds {', '.join(map(str, SEEDS))} (max - min), max per regime/cutoff:")
    print(spread.groupby(level=["cutoff", "regime"]).max().round(1).to_string())


def main() -> None:
    games = all_games()
    rows = []
    for regime in REGIMES:
        for cutoff in ("std", "1pc"):
            for seed in SEEDS:
                rows += rate(games, regime, cutoff, seed)
        rows += rate(games, regime, "1pcruns", ELO_SEED)
    ratings = pd.DataFrame(rows)
    ratings.to_csv(output_path(OUT_DATA, "elo_all.csv"), index=False)

    pd.set_option("display.width", 250)
    print_ratings(ratings)


if __name__ == "__main__":
    main()
