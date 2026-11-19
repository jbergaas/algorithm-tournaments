"""The matches of one network and budget, read from the batch outputs and grouped by condition
(threshold, sequence, algorithm A, algorithm B), and every check the analysis relies on:

- each batch holds exactly the planned matches, and the ownership counts add up to n;
- the runs of a pair without Random all give the same outcome;
- a pair's candidate lists and RNG seeds in a run are the same at every threshold and under
  every sequence, and the seeds a sequence hands out are the same at every threshold;
- Random's candidates in a run depend only on the run and the player, not on the opponent.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import design

INTEGER_COLUMNS = ('nodes', 'seed_count', 'run_id', 'rng_seed_a', 'rng_seed_b', 'spread_a', 'spread_b',
                   'neutral', 'ranking_overlap', 'skipped_a', 'skipped_b', 'rounds')
LIST_COLUMNS = ('ranking_a', 'ranking_b', 'seeds_a', 'seeds_b')
MATCH_KEY = ('algorithm_a', 'algorithm_b', 'seed_count', 'run_id', 'draft_order')
OUTCOME = ('spread_a', 'spread_b', 'neutral', 'ranking_overlap', 'skipped_a', 'skipped_b', 'rounds',
           'stop_reason', *LIST_COLUMNS)
CANDIDATES = ('ranking_a', 'ranking_b', 'rng_seed_a', 'rng_seed_b')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def with_random(match: dict) -> bool:
    return design.with_random(match['algorithm_a'], match['algorithm_b'])


def condition_key(match: dict) -> tuple:
    return (match['theta'], match['allocation_order'], match['algorithm_a'], match['algorithm_b'])


def read_batch(folder: Path, k: int) -> list[dict]:
    """One match per row of results.csv, joined with the candidate lists and seeds of the same
    row of rankings.csv (the engine writes both files in the same match order)."""
    matches = []
    with open(folder / 'results.csv', newline='') as results, open(folder / 'rankings.csv', newline='') as rankings:
        for result, trace in zip(csv.DictReader(results), csv.DictReader(rankings), strict=True):
            require(all(result[column] == trace[column] for column in MATCH_KEY),
                    f'{folder}: results.csv and rankings.csv are not in the same order')
            match = {column: int(result[column]) for column in INTEGER_COLUMNS}
            match.update({column: json.loads(trace[column]) for column in LIST_COLUMNS})
            match.update(algorithm_a=result['algorithm_a'], algorithm_b=result['algorithm_b'],
                         allocation_order=result['draft_order'], theta=float(result['theta']), k=k,
                         stop_reason=result['stop_reason'])
            matches.append(match)
    return matches


def check_batch(folder: Path, matches: list[dict], config: dict) -> None:
    played = [(m['algorithm_a'], m['algorithm_b'], m['allocation_order'], m['run_id']) for m in matches]
    require(len(played) == len(set(played)) and set(played) == design.planned_matches(config),
            f'{folder}: the matches differ from the planned ones')
    for match in matches:
        require(match['spread_a'] + match['spread_b'] + match['neutral'] == match['nodes'],
                f'{folder}: ownership counts do not add up to the number of nodes')


def check_deterministic_runs(groups: dict) -> None:
    for key, rows in groups.items():
        if not with_random(rows[0]):
            outcomes = {tuple(json.dumps(row[field]) for field in OUTCOME) for row in rows}
            require(len(outcomes) == 1, f'runs of a pair without Random gave different outcomes: {key}')


def check_candidates(matches: list[dict]) -> None:
    by_run = defaultdict(list)
    for match in matches:
        by_run[(match['algorithm_a'], match['algorithm_b'], match['run_id'])].append(match)
    for rows in by_run.values():
        first = rows[0]
        require(all(row[field] == first[field] for row in rows for field in CANDIDATES),
                'candidate lists or seeds differ between thresholds or sequences')
        seeds = {}
        for row in rows:
            pair = seeds.setdefault(row['allocation_order'], (row['seeds_a'], row['seeds_b']))
            require(pair == (row['seeds_a'], row['seeds_b']), 'the seeds of a sequence differ between thresholds')
    random_draws = {}
    for match in matches:
        for side in ('a', 'b'):
            if match['algorithm_' + side] == 'random':
                draw = (match['rng_seed_' + side], match['ranking_' + side])
                require(random_draws.setdefault((side, match['run_id']), draw) == draw,
                        'Random candidates differ between opponents')


def read_partition(run_root: Path, config: dict, network: str, k: int) -> dict[tuple, list[dict]]:
    """All matches of one network and budget, checked and grouped by condition. Within a group
    the runs are in run order."""
    matches = []
    for theta in config['thresholds']:
        folder = design.batch_dir(run_root, network, k, theta)
        batch = read_batch(folder, k)
        check_batch(folder, batch, config)
        matches.extend(batch)
    groups = defaultdict(list)
    for match in matches:
        groups[condition_key(match)].append(match)
    check_deterministic_runs(groups)
    check_candidates(matches)
    return dict(groups)
