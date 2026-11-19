#!/usr/bin/env python3
"""Analysis of one network and budget: reads its batches (one per threshold) and writes, with
self-play (A and B the same algorithm) in separate files:

    condition_summaries_cross_play.csv, self_play_diagnostics.csv        one row per condition
    paired_contrasts_cross_play.csv, self_play_paired_contrasts.csv      sequence contrasts with
                                                                         sequential_a
    overlap_conditions_cross_play.csv, overlap_conditions_self_play.csv  candidate overlap and
    overlap_contrasts_cross_play.csv, overlap_contrasts_self_play.csv    seed changes
    condition_uncertainty.csv, self_play_uncertainty.csv,                bootstrap intervals
    paired_contrast_uncertainty.csv, self_play_paired_uncertainty.csv    (Random)
    figures/                                                             diagnostic figures

analysis/combined/ holds the same tables for all networks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bootstrap
import conditions
import overlap
import partition_figures
from records import read_partition
from tables import write_csv

UNCERTAINTY_COLUMNS = ['graph_name', 'k', 'theta', 'allocation_order', 'algorithm_a', 'algorithm_b',
                       'is_self_play', 'random_involving', 'raw_execution_count', 'bootstrap_block_id', 'mean',
                       'sample_standard_deviation', 'interval_lower_95', 'interval_upper_95',
                       'empirical_bootstrap_degenerate', 'constant_sample', 'scientific_realization_count']
CONTRAST_UNCERTAINTY_COLUMNS = (UNCERTAINTY_COLUMNS[:3] + UNCERTAINTY_COLUMNS[4:6]
                                + ['reference_order', 'comparison_order'] + UNCERTAINTY_COLUMNS[6:])

# (cross-play file, self-play file) of each table.
FILES = {
    'summaries': ('condition_summaries_cross_play.csv', 'self_play_diagnostics.csv'),
    'contrasts': ('paired_contrasts_cross_play.csv', 'self_play_paired_contrasts.csv'),
    'overlap': ('overlap_conditions_cross_play.csv', 'overlap_conditions_self_play.csv'),
    'overlap_contrasts': ('overlap_contrasts_cross_play.csv', 'overlap_contrasts_self_play.csv'),
    'intervals': ('condition_uncertainty.csv', 'self_play_uncertainty.csv'),
    'contrast_intervals': ('paired_contrast_uncertainty.csv', 'self_play_paired_uncertainty.csv'),
}


def write_split(output: Path, table: str, rows: list[dict], columns: list[str] | None = None) -> None:
    cross_play, self_play = FILES[table]
    write_csv(output / cross_play, [row for row in rows if not row['is_self_play']], columns)
    write_csv(output / self_play, [row for row in rows if row['is_self_play']], columns)


def analyse(run_root: Path, network: str, k: int, output: Path) -> None:
    config = json.loads((run_root / 'config.json').read_text())
    groups = read_partition(run_root, config, network, k)
    summaries, contrasts = conditions.analyse(network, k, groups, config)
    overlap_rows, overlap_contrasts = overlap.analyse(network, groups, config)
    intervals, contrast_intervals = bootstrap.analyse(network, k, groups, config)
    del groups

    output.mkdir(parents=True)
    write_split(output, 'summaries', summaries)
    write_split(output, 'contrasts', contrasts)
    write_split(output, 'overlap', overlap_rows)
    write_split(output, 'overlap_contrasts', overlap_contrasts)
    write_split(output, 'intervals', intervals, UNCERTAINTY_COLUMNS)
    write_split(output, 'contrast_intervals', contrast_intervals, CONTRAST_UNCERTAINTY_COLUMNS)
    partition_figures.draw(output / 'figures', network, k, config, summaries, contrasts, overlap_rows,
                           contrast_intervals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--network', required=True)
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyse(args.run_root, args.network, args.budget, args.output)


if __name__ == '__main__':
    main()
