"""Elo ratings averaged over random game orders, in the two forms used by the analyses.

Every game moves both ratings by K (s - e) with e = 1 / (1 + 10^((r_opponent - r) / 400)), K = 16,
everyone starting at 1500. The forms differ only in how the random game orders are drawn:
  elo_variants  argsort of uniform numbers, one fresh generator per call (the rating variants and
                the generated networks)
  elo_table     consecutive permutations from a generator shared by the networks of one table
                column (the ratings of the result tables)
"""
import sys
from pathlib import Path

import numba as nb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared  # noqa: E402,F401  (numba's cache folder)

K_FACTOR = 16.0
VARIANT_SEED = 20261105


@nb.njit(cache=True)
def _mean_over_orders(a, b, score_b, orders, n_players, k_factor):
    total = np.zeros(n_players)
    for o in range(orders.shape[0]):
        r = np.full(n_players, 1500.0)
        for t in range(orders.shape[1]):
            i = orders[o, t]
            ra = r[a[i]]
            rb = r[b[i]]
            delta = k_factor * (score_b[i] - 1.0 / (1.0 + 10.0 ** ((ra - rb) / 400.0)))
            r[b[i]] += delta
            r[a[i]] -= delta
        total += r
    return total / orders.shape[0]


def elo_variants(a: np.ndarray, b: np.ndarray, score_b: np.ndarray, orders: int = 100,
                 seed: int = VARIANT_SEED, n_players: int = 4) -> np.ndarray:
    """Mean final rating of each player; a and b index the players of each game, score_b is B's score."""
    if len(a) == 0:
        return np.full(n_players, np.nan)
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((orders, len(a))), axis=1)
    return _mean_over_orders(a.astype(np.int64), b.astype(np.int64), score_b.astype(np.float64),
                             order.astype(np.int64), n_players, K_FACTOR)


def elo_table(a: np.ndarray, b: np.ndarray, score_b: np.ndarray, rng: np.random.Generator, orders: int = 200,
              n_players: int = 5) -> np.ndarray:
    """Mean final rating over `orders` permutations drawn one after the other from rng."""
    if len(a) == 0:
        return np.full(n_players, np.nan)
    perm = np.stack([rng.permutation(len(a)) for _ in range(orders)])
    r = np.full((orders, n_players), 1500.0)
    rows = np.arange(orders)
    for t in range(len(a)):
        i = perm[:, t]
        ai, bi = a[i], b[i]
        expected_b = 1 / (1 + 10 ** ((r[rows, ai] - r[rows, bi]) / 400))
        delta = K_FACTOR * (score_b[i] - expected_b)
        r[rows, bi] += delta
        r[rows, ai] -= delta
    total = np.zeros(n_players)
    for o in range(orders):
        total += r[o]
    return total / orders


def mean_scores(a: np.ndarray, b: np.ndarray, score_b: np.ndarray, n_players: int = 4) -> np.ndarray:
    """Mean score of each player over all its games, in both roles."""
    total = (np.bincount(b, weights=score_b, minlength=n_players)
             + np.bincount(a, weights=1 - score_b, minlength=n_players))
    count = np.bincount(b, minlength=n_players) + np.bincount(a, minlength=n_players)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count
