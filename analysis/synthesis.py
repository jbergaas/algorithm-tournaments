#!/usr/bin/env python3
"""Combines the tables of every network and budget into analysis/combined/: each table of
partition.py, concatenated, plus

    cross_network_condition_values.csv   each condition's mean margin on every network, and
                                         whether its sign differs between networks
    network_coverage.csv                 conditions and matches per network
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import design
from partition import FILES
from tables import read_csv, write_csv

CONDITION_KEY = ('k', 'theta', 'allocation_order', 'algorithm_a', 'algorithm_b')


def partitions(run_root: Path, config: dict) -> list[Path]:
    return [run_root / 'analysis' / network['name'] / f'k{k}'
            for network in config['networks'] for k in config['budgets']]


def concatenate(sources: list[Path], name: str, target: Path) -> None:
    with open(target, 'w', newline='') as output:
        writer = csv.writer(output)
        for index, folder in enumerate(sources):
            with open(folder / name, newline='') as stream:
                reader = csv.reader(stream)
                header = next(reader)
                if index == 0:
                    writer.writerow(header)
                writer.writerows(reader)


def coverage_and_signs(sources: list[Path], output: Path) -> dict:
    """Writes network_coverage.csv and cross_network_condition_values.csv; returns the totals."""
    coverage = defaultdict(lambda: dict(conditions=0, executions=0, condition_realizations=0))
    by_condition = defaultdict(list)
    for folder in sources:
        for name in FILES['summaries']:
            for row in read_csv(folder / name):
                counts = coverage[row['graph_name']]
                counts['conditions'] += 1
                counts['executions'] += int(row['raw_execution_count'])
                counts['condition_realizations'] += int(row['scientific_realization_count'])
                key = (int(row['k']), float(row['theta'])) + tuple(row[field] for field in CONDITION_KEY[2:])
                by_condition[key].append((row['graph_name'], float(row['mean_margin_b'])))
    write_csv(output / 'network_coverage.csv', [dict(graph_name=name, **counts) for name, counts in coverage.items()])
    rows = []
    for key, values in by_condition.items():
        signs = {(margin > 0) - (margin < 0) for _, margin in values}
        for network, margin in values:
            rows.append(dict(zip(CONDITION_KEY, key), graph_name=network, mean_margin_b=margin,
                             matched_network_count=len(values), observed_cross_network_sign_reversal={-1, 1} <= signs))
    write_csv(output / 'cross_network_condition_values.csv', rows)
    return {field: sum(counts[field] for counts in coverage.values())
            for field in ('conditions', 'executions', 'condition_realizations')}


def synthesize(run_root: Path, output: Path) -> None:
    config = json.loads((run_root / 'config.json').read_text())
    sources = partitions(run_root, config)
    output.mkdir(parents=True)
    for names in FILES.values():
        for name in names:
            concatenate(sources, name, output / name)
    found, expected = coverage_and_signs(sources, output), design.totals(config)
    if found != expected:
        raise ValueError(f'the partitions hold {found} instead of {expected}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    synthesize(args.run_root, args.output)


if __name__ == '__main__':
    main()
