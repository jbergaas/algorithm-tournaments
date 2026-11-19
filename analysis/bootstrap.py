"""Bootstrap intervals for the conditions with Random.

Random draws new candidates in every run, so a condition with Random has one margin per run.
Its interval is a pointwise 95% percentile bootstrap over the runs: 10,000 resamples of the run
IDs with replacement, from NumPy's PCG64. All conditions and contrasts of one network and budget
use the same resampled run IDs (one block), because the same Random candidates enter all of
them in a given run. The block's generator is seeded from the bootstrap seed and the SHA-256 of
the block name. Conditions without Random have a single exact value and no interval.
"""
from __future__ import annotations

import hashlib
from statistics import fmean, stdev

import numpy as np

from conditions import margin
from records import with_random

BOOTSTRAP_SEED = 20261014
RESAMPLES = 10000


def resample_counts(sample_size: int, resamples: int, seed: int, block_id: str) -> np.ndarray:
    """How often each run appears in each resample: a resamples x sample_size matrix."""
    entropy = np.frombuffer(hashlib.sha256(block_id.encode()).digest(), dtype='<u4').tolist()
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, *entropy])))
    indices = generator.integers(0, sample_size, size=(resamples, sample_size), dtype=np.int64)
    counts = np.zeros((resamples, sample_size), dtype=np.int64)
    np.add.at(counts, (np.arange(resamples)[:, None], indices), 1)
    return counts


def interval_columns(vectors: list[np.ndarray], counts: np.ndarray) -> list[dict]:
    """Mean, standard deviation and 95% percentile interval of each vector of per-run values."""
    values = np.column_stack(vectors)
    distribution = counts @ values / values.shape[0]
    constant = np.all(values == values[0, :], axis=0)
    distribution[:, constant] = values[0, constant]
    limits = np.quantile(distribution, [.025, .975], axis=0, method='linear')
    return [dict(mean=fmean(vector), sample_standard_deviation=stdev(vector),
                 interval_lower_95=float(limits[0, index]), interval_upper_95=float(limits[1, index]),
                 empirical_bootstrap_degenerate=bool(np.all(distribution[:, index] == distribution[0, index])),
                 constant_sample=bool(np.all(vector == vector[0])), scientific_realization_count=len(vector))
            for index, vector in enumerate(vectors)]


def exact_value(value: float) -> dict:
    return dict(mean=value, sample_standard_deviation=None, interval_lower_95=None, interval_upper_95=None,
                empirical_bootstrap_degenerate=None, constant_sample=None, scientific_realization_count=1)


def margins(rows: list[dict]) -> np.ndarray:
    """B's margin in each run, in run order."""
    return np.array([margin(row) for row in rows], dtype=np.float64)


def analyse(network: str, k: int, groups: dict, config: dict) -> tuple[list[dict], list[dict]]:
    """Interval rows of every condition and of every contrast with sequential_a, sorted by
    threshold, sequence and algorithm names."""
    block_id = f'{network}/k{k}'
    others = [sequence for sequence in config['sequences'] if sequence != 'sequential_a']
    conditions, contrasts = [], []
    bootstrapped, vectors = [], []
    for key in sorted(groups):
        theta, sequence, a, b = key
        rows = groups[key]
        random = with_random(rows[0])
        base = dict(graph_name=network, k=k, theta=theta, allocation_order=sequence, algorithm_a=a,
                    algorithm_b=b, is_self_play=a == b, random_involving=random, raw_execution_count=len(rows),
                    bootstrap_block_id=block_id if random else None)
        values = margins(rows)
        conditions.append(base)
        if random:
            bootstrapped.append(base)
            vectors.append(values)
        else:
            base.update(exact_value(float(values[0])))
        if sequence != 'sequential_a':
            continue
        for other in others:
            differences = margins(groups[(theta, other, a, b)]) - values
            row = {key: value for key, value in base.items() if key != 'allocation_order'}
            row.update(reference_order='sequential_a', comparison_order=other)
            contrasts.append(row)
            if random:
                bootstrapped.append(row)
                vectors.append(differences)
            else:
                row.update(exact_value(float(differences[0])))
    if vectors:
        counts = resample_counts(config['runs'], RESAMPLES, BOOTSTRAP_SEED, block_id)
        for row, interval in zip(bootstrapped, interval_columns(vectors, counts)):
            row.update(interval)
    return conditions, contrasts
