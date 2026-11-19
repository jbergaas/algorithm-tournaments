"""Condition summaries and sequence contrasts of one network and budget.

A condition is a threshold, a picking sequence and an ordered pair of algorithms (A, B). With
Random in the pair, each of its runs is one value; without Random every run gives the same outcome
(records.py checks this), and the condition counts once. Margins are B-positive:
(B's nodes - A's nodes) / n.
"""
from __future__ import annotations

from collections import Counter
from statistics import fmean

import design
from records import condition_key, with_random


def margin(match: dict) -> float:
    return (match['spread_b'] - match['spread_a']) / match['nodes']


def counted_runs(rows: list[dict]) -> list[dict]:
    """The runs that count as values: all of them with Random, else the first."""
    return rows if with_random(rows[0]) else rows[:1]


def summarize(network: str, k: int, rows: list[dict]) -> dict:
    theta, sequence, a, b = condition_key(rows[0])
    values = counted_runs(rows)
    outcomes = [(row['spread_b'] > row['spread_a']) - (row['spread_b'] < row['spread_a']) for row in values]
    summary = dict(
        graph_name=network, k=k, theta=theta, allocation_order=sequence, algorithm_a=a, algorithm_b=b,
        random_involving=with_random(rows[0]), is_self_play=a == b,
        raw_execution_count=len(rows), scientific_realization_count=len(values),
        mean_a_fraction=fmean(row['spread_a'] / row['nodes'] for row in values),
        mean_b_fraction=fmean(row['spread_b'] / row['nodes'] for row in values),
        mean_neutral_fraction=fmean(row['neutral'] / row['nodes'] for row in values),
        mean_margin_b=fmean(margin(row) for row in values),
        mean_acquisition_rounds=fmean(row['rounds'] for row in values),
        mean_ranking_overlap=fmean(row['ranking_overlap'] for row in values),
        mean_skipped_a=fmean(row['skipped_a'] for row in values),
        mean_skipped_b=fmean(row['skipped_b'] for row in values),
        strict_a_win_proportion=outcomes.count(-1) / len(values),
        strict_b_win_proportion=outcomes.count(1) / len(values),
        draw_proportion=outcomes.count(0) / len(values),
        stop_reason_counts=dict(Counter(row['stop_reason'] for row in rows)))
    summary['draw_adjusted_score_a'] = summary['strict_a_win_proportion'] + .5 * summary['draw_proportion']
    summary['draw_adjusted_score_b'] = summary['strict_b_win_proportion'] + .5 * summary['draw_proportion']
    return summary


def contrast(network: str, k: int, reference: list[dict], comparison: list[dict]) -> dict:
    """Mean change of B's margin from sequential_a to another sequence, paired by run."""
    differences = [margin(other) - margin(row) for row, other in zip(reference, comparison)]
    counted = differences if with_random(reference[0]) else differences[:1]
    theta, _, a, b = condition_key(reference[0])
    return dict(graph_name=network, k=k, theta=theta, algorithm_a=a, algorithm_b=b,
                reference_order='sequential_a', comparison_order=comparison[0]['allocation_order'],
                random_involving=with_random(reference[0]), is_self_play=a == b,
                paired_execution_count=len(differences), scientific_realization_count=len(counted),
                mean_delta_margin_b=fmean(counted))


def analyse(network: str, k: int, groups: dict, config: dict) -> tuple[list[dict], list[dict]]:
    """Summaries of every condition (by threshold, sequence and pair) and the contrasts of every
    other sequence with sequential_a (by threshold, pair and sequence)."""
    others = [sequence for sequence in config['sequences'] if sequence != 'sequential_a']
    summaries, contrasts = [], []
    for theta in map(float, config['thresholds']):
        for sequence in config['sequences']:
            summaries.extend(summarize(network, k, groups[(theta, sequence, a, b)]) for a, b in design.pairs(config))
        for a, b in design.pairs(config):
            for sequence in others:
                contrasts.append(contrast(network, k, groups[(theta, 'sequential_a', a, b)],
                                          groups[(theta, sequence, a, b)]))
    return summaries, contrasts
