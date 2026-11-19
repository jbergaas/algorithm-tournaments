"""Networks, variants and the variant diffusion of the sensitivity check.

Each variant changes one assumption of the diffusion model:
  base      the model of the study: one threshold theta for every node, a tied node stays neutral
  hom-1%, hom+1%
            every node has the threshold 0.99 theta or 1.01 theta (a knife-edge check)
  U0.01 .. U0.50
            node v has the threshold theta (1 + w U_v), U_v uniform on [-1, 1], kept in [0, 1]
  D3        node v has the threshold theta / 2, theta or 3 theta / 2 with probability 1/3 each
            (U_v < -1/3, |U_v| <= 1/3, U_v > 1/3)
  tie       one threshold theta; a neutral node with as many owned neighbours of each player,
            enough to meet its threshold, joins one of them by a fair coin
U_v is drawn once per network and draw and reused for every w, budget and threshold, so the
uniform variants differ only in width. Each drawn variant is sampled 20 times.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402  (before numba: it sets numba's cache folder)

import numba as nb  # noqa: E402
import numpy as np  # noqa: E402

# A network's position in NETWORKS enters the seeds of its random draws and tie coins, so this
# order fixes the results.
REAL = ["ca-GrQc", "Erdos02", "wiki-Vote", "ca-HepPh", "email-Enron", "com-DBLP", "ego-Facebook",
        "soc-Epinions1", "ca-HepTh", "ca-CondMat", "com-Youtube"]
GENERATED = ["er-d8-i0", "cl-g2.5-i0", "ws-p0.01-i0", "collab-s50-i0"]
NETWORKS = REAL + GENERATED
BUDGETS = (10, 20)
THETAS = s.THETAS
# 16 ordered pairs (12 of different algorithms, 4 self-play) x 4 sequences = 64 matches per unit
MATCHES = [(a, b, q) for a in s.STRUCTURAL for b in s.STRUCTURAL for q in s.SEQUENCES]
DRAWS = 20
BASE_SEED = 20261105
# name, kind, parameter, number of draws
VARIANTS = [("base", "hom", 1.0, 1), ("hom-1%", "hom", 0.99, 1), ("hom+1%", "hom", 1.01, 1),
            ("U0.01", "unif", 0.01, DRAWS), ("U0.10", "unif", 0.10, DRAWS),
            ("U0.25", "unif", 0.25, DRAWS), ("U0.50", "unif", 0.50, DRAWS),
            ("D3", "disc", 0.5, DRAWS), ("tie", "tie", 0.0, DRAWS)]
VARIANT_NAMES = [v[0] for v in VARIANTS]

NEUTRAL, PLAYER_A, PLAYER_B = 0, 1, 2
GOLD = np.uint64(0x9E3779B97F4A7C15)
ROUND_KEY = np.uint64(0xD6E8FEB86659FD93)


def uniform_draw(network: str, n: int, draw: int) -> np.ndarray:
    """U_v of one network and draw."""
    rng = np.random.default_rng([BASE_SEED, NETWORKS.index(network), draw])
    return rng.uniform(-1.0, 1.0, n)


def node_thresholds(kind: str, param: float, theta: float, u: np.ndarray | None, n: int) -> np.ndarray:
    """Threshold of every node under one variant and draw."""
    if kind == "hom":
        return np.full(n, theta * param)
    if kind == "tie":
        return np.full(n, theta)
    if kind == "unif":
        factor = 1.0 + param * u
    else:
        factor = np.where(u < -1 / 3, 1.0 - param, np.where(u > 1 / 3, 1.0 + param, 1.0))
    return np.clip(theta * factor, 0.0, 1.0)


def match_seed(*parts: int) -> np.uint64:
    """64-bit seed of the tie coin of one match, from integers."""
    x = 0x243F6A8885A308D3
    for p in parts:
        x = (x ^ (p & 0xFFFFFFFFFFFFFFFF)) * 0x100000001B3 & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 29
    return np.uint64(x)


@nb.njit(cache=True, inline="always")
def _mix64(x):
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


@nb.njit(cache=True)
def propagate(indptr, indices, owner, thr, tie_coin, seed, max_rounds):
    """The diffusion of shared._propagate with a threshold per node: node v joins a player with
    strictly more owned neighbours once that count is at least thr[v] * degree. With tie_coin, a
    node whose two counts are equal, at least 1 and enough for its threshold joins one player by a
    coin, a hash of (seed, node, round). owner is changed in place. Returns the rounds, the stop
    reason code and the number of nodes left neutral only because of a tie."""
    n = owner.shape[0]
    count_a = np.zeros(n, np.int32)
    count_b = np.zeros(n, np.int32)
    in_frontier = np.zeros(n, np.uint8)
    frontier = np.empty(n, np.int64)
    following = np.empty(n, np.int64)
    changed = np.empty(n, np.int64)
    changed_owner = np.empty(n, np.int8)
    size = 0
    for u in range(n):
        o = owner[u]
        if o == NEUTRAL:
            continue
        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]
            if owner[v] != NEUTRAL:
                continue
            if o == PLAYER_A:
                count_a[v] += 1
            else:
                count_b[v] += 1
            if in_frontier[v] == 0:
                in_frontier[v] = 1
                frontier[size] = v
                size += 1
    rounds = 0
    stop = 2
    while rounds < max_rounds:
        if size == 0:
            stop = 0
            break
        n_changed = 0
        for i in range(size):
            u = frontier[i]
            in_frontier[u] = 0
            if owner[u] != NEUTRAL:
                continue
            need = thr[u] * float(indptr[u + 1] - indptr[u])
            ca = count_a[u]
            cb = count_b[u]
            if ca > cb and float(ca) >= need:
                changed[n_changed] = u
                changed_owner[n_changed] = PLAYER_A
                n_changed += 1
            elif cb > ca and float(cb) >= need:
                changed[n_changed] = u
                changed_owner[n_changed] = PLAYER_B
                n_changed += 1
            elif tie_coin and ca == cb and ca >= 1 and float(ca) >= need:
                h = _mix64(seed ^ (np.uint64(u) * GOLD) ^ (np.uint64(rounds + 1) * ROUND_KEY))
                changed[n_changed] = u
                changed_owner[n_changed] = PLAYER_A if (h & np.uint64(1)) == np.uint64(0) else PLAYER_B
                n_changed += 1
        if n_changed == 0:
            stop = 1
            break
        for i in range(n_changed):
            owner[changed[i]] = changed_owner[i]
        next_size = 0
        for i in range(n_changed):
            u = changed[i]
            o = changed_owner[i]
            for p in range(indptr[u], indptr[u + 1]):
                v = indices[p]
                if owner[v] != NEUTRAL:
                    continue
                if o == PLAYER_A:
                    count_a[v] += 1
                else:
                    count_b[v] += 1
                if in_frontier[v] == 0:
                    in_frontier[v] = 1
                    following[next_size] = v
                    next_size += 1
        frontier, following = following, frontier
        size = next_size
        rounds += 1
    tied = 0
    for u in range(n):
        if owner[u] == NEUTRAL and count_a[u] == count_b[u] and count_a[u] >= 1:
            if float(count_a[u]) >= thr[u] * float(indptr[u + 1] - indptr[u]):
                tied += 1
    return rounds, stop, tied
