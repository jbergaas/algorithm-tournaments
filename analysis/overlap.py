"""How much the two candidate lists overlap, and what changing the picking sequence changes.

Before any seed is placed, each player's algorithm lists 2k candidates, which depend only on the
pair and the run. Per condition this module reports the overlap of the two lists, whole and in the
first k entries. Per sequence contrast with sequential_a it reports how often the seeds, the
ownership and the winner change. The conditional means use only the runs whose two top-k lists
share at least one node.
"""
from __future__ import annotations

from statistics import fmean

from conditions import counted_runs, margin
from records import condition_key, with_random

CHANGES = ('seed_a_changed', 'seed_b_changed', 'seed_pair_changed', 'ownership_changed',
           'margin_changed', 'winner_changed', 'rounds_changed')


def seed_pair(match: dict) -> list:
    return [sorted(match['seeds_a']), sorted(match['seeds_b'])]


def winner(match: dict) -> int:
    return (match['spread_b'] > match['spread_a']) - (match['spread_b'] < match['spread_a'])


def run_overlaps(groups: dict, sequences: list[str]) -> dict[tuple, dict]:
    """Per (algorithm A, algorithm B, run): the overlap of the two candidate lists, and whether
    any sequence hands out other seeds than sequential_a. Both are the same at every threshold
    (records.py checks this), so the smallest threshold is used."""
    theta = min(key[0] for key in groups)
    overlaps = {}
    for (row_theta, sequence, a, b), rows in groups.items():
        if row_theta != theta or sequence != 'sequential_a':
            continue
        for run, row in enumerate(rows):
            k = row['k']
            seed_pairs = [seed_pair(groups[(theta, other, a, b)][run]) for other in sequences]
            overlaps[(a, b, run)] = dict(
                full_2k_overlap=len(set(row['ranking_a']) & set(row['ranking_b'])),
                top_k_overlap=len(set(row['ranking_a'][:k]) & set(row['ranking_b'][:k])),
                any_sequence_seed_pair_differs=any(pair != seed_pair(row) for pair in seed_pairs))
    return overlaps


def condition_row(network: str, rows: list[dict], overlaps: dict) -> dict:
    theta, sequence, a, b = condition_key(rows[0])
    values = counted_runs(rows)
    infos = [overlaps[(a, b, row['run_id'])] for row in values]
    k, nodes = rows[0]['k'], rows[0]['nodes']
    return dict(
        graph_name=network, k=k, theta=theta, allocation_order=sequence, algorithm_a=a, algorithm_b=b,
        nodes=nodes, random_involving=with_random(rows[0]), is_self_play=a == b,
        raw_execution_count=len(rows), scientific_realization_count=len(values),
        top_k_overlap_eligible_count=sum(info['top_k_overlap'] > 0 for info in infos),
        mean_full_2k_overlap=fmean(info['full_2k_overlap'] for info in infos),
        mean_top_k_overlap=fmean(info['top_k_overlap'] for info in infos),
        any_sequence_seed_pair_differs_proportion=fmean(info['any_sequence_seed_pair_differs'] for info in infos),
        mean_margin_b=fmean(margin(row) for row in values),
        mean_absolute_margin=fmean(abs(margin(row)) for row in values),
        mean_neutral_fraction=fmean(row['neutral'] / row['nodes'] for row in values),
        mean_acquired_beyond_seeds=fmean(row['spread_a'] + row['spread_b'] - 2 * k for row in values),
        mean_acquired_beyond_seeds_fraction=fmean((row['spread_a'] + row['spread_b'] - 2 * k) / row['nodes']
                                                  for row in values),
        mean_spread_a=fmean(row['spread_a'] for row in values),
        mean_spread_b=fmean(row['spread_b'] for row in values),
        mean_neutral_count=fmean(row['neutral'] for row in values),
        mean_acquisition_rounds=fmean(row['rounds'] for row in values))


def compare_runs(reference: dict, comparison: dict) -> dict:
    k = reference['k']
    top_k_overlap = len(set(reference['ranking_a'][:k]) & set(reference['ranking_b'][:k]))
    return dict(
        run_id=reference['run_id'], eligible=top_k_overlap > 0,
        delta_margin_b=margin(comparison) - margin(reference),
        seed_a_changed=sorted(reference['seeds_a']) != sorted(comparison['seeds_a']),
        seed_b_changed=sorted(reference['seeds_b']) != sorted(comparison['seeds_b']),
        seed_pair_changed=seed_pair(reference) != seed_pair(comparison),
        ownership_changed=any(reference[x] != comparison[x] for x in ('spread_a', 'spread_b', 'neutral')),
        margin_changed=margin(reference) != margin(comparison),
        winner_changed=winner(reference) != winner(comparison),
        rounds_changed=reference['rounds'] != comparison['rounds'])


def mean_or_none(values: list):
    return fmean(values) if values else None


def contrast_row(network: str, reference: list[dict], comparison: list[dict]) -> dict:
    details = [compare_runs(row, other) for row, other in zip(reference, comparison)]
    counted = details if with_random(reference[0]) else details[:1]
    eligible = [detail for detail in counted if detail['eligible']]
    theta, _, a, b = condition_key(reference[0])
    row = dict(
        graph_name=network, k=reference[0]['k'], theta=theta, algorithm_a=a, algorithm_b=b,
        reference_order='sequential_a', comparison_order=comparison[0]['allocation_order'],
        random_involving=with_random(reference[0]), is_self_play=a == b,
        total_executions=len(details), eligible_executions=sum(detail['eligible'] for detail in details),
        total_scientific_realizations=len(counted), eligible_scientific_realizations=len(eligible),
        eligible_run_ids=[detail['run_id'] for detail in eligible],
        mean_delta_margin_b_unconditional=fmean(detail['delta_margin_b'] for detail in counted),
        mean_delta_margin_b_conditional_top_k_overlap=mean_or_none([d['delta_margin_b'] for d in eligible]))
    for change in CHANGES:
        row[change + '_proportion_unconditional'] = fmean(detail[change] for detail in counted)
        row[change + '_proportion_conditional'] = mean_or_none([detail[change] for detail in eligible])
    return row


def analyse(network: str, groups: dict, config: dict) -> tuple[list[dict], list[dict]]:
    """Rows sorted by threshold, sequence and algorithm names."""
    overlaps = run_overlaps(groups, config['sequences'])
    conditions = [condition_row(network, groups[key], overlaps) for key in sorted(groups)]
    others = [sequence for sequence in config['sequences'] if sequence != 'sequential_a']
    contrasts = [contrast_row(network, groups[(theta, 'sequential_a', a, b)], groups[(theta, other, a, b)])
                 for theta, sequence, a, b in sorted(groups, key=lambda key: (key[0], key[2], key[3]))
                 if sequence == 'sequential_a' for other in others]
    return conditions, contrasts
