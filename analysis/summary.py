#!/usr/bin/env python3
"""Writes reports/summary.md: what the run contains, and per network how many matchup conditions
left almost every node neutral and how many sequence contrasts were exactly zero."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tables import read_csv


def per_network_lines(combined: Path) -> list[str]:
    conditions, neutral = Counter(), Counter()
    for row in read_csv(combined / 'condition_summaries_cross_play.csv'):
        conditions[row['graph_name']] += 1
        neutral[row['graph_name']] += float(row['mean_neutral_fraction']) >= .95
    contrasts, zero = Counter(), Counter()
    for row in read_csv(combined / 'paired_contrasts_cross_play.csv'):
        contrasts[row['graph_name']] += 1
        zero[row['graph_name']] += float(row['mean_delta_margin_b']) == 0
    return [f'- {network}: {conditions[network]:,} matchup conditions, {neutral[network]:,} with a mean neutral '
            f'fraction of at least 0.95; {zero[network]:,} of {contrasts[network]:,} sequence contrasts exactly zero'
            for network in conditions]


def summary(run_root: Path) -> str:
    config = json.loads((run_root / 'config.json').read_text())
    coverage = list(read_csv(run_root / 'analysis/combined/network_coverage.csv'))
    matches = sum(int(row['executions']) for row in coverage)
    conditions = sum(int(row['conditions']) for row in coverage)
    lines = [
        '# Run summary',
        '',
        f"- Networks ({len(config['networks'])}): {', '.join(n['name'] for n in config['networks'])}",
        f"- Algorithms: {', '.join(config['algorithms'])}",
        f"- Picking sequences: {', '.join(config['sequences'])}",
        f"- Budgets k: {', '.join(map(str, config['budgets']))}",
        f"- Thresholds: {', '.join(f'{theta:g}' for theta in config['thresholds'])}",
        f"- Runs: {config['runs']} for pairs with Random, {config['deterministic_runs']} for the other pairs",
        f'- Matches played: {matches:,}; conditions: {conditions:,}',
        f"- Random's seeds from base seed {config['base_seed']}",
        f"- Exact Betweenness on {config['betweenness_threads']} threads",
        '',
        'Per network (cross-play only):',
        '',
        *per_network_lines(run_root / 'analysis/combined'),
        '',
        'The tables are in analysis/combined/ and the matchup figures in figures/.',
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    (args.output / 'summary.md').write_text(summary(args.run_root))


if __name__ == '__main__':
    main()
