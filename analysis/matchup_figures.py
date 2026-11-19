#!/usr/bin/env python3
"""One matchup heatmap per network: the mean margin (B - A) / n of every ordered pair of
algorithms at the largest budget, the threshold closest to 1/4 and sequential_a. Writes
<network>-matchup.png, .pdf and .csv (the plotted values) for each network."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from partition import FILES
from partition_figures import pair_heatmap, plt
from tables import read_csv, write_csv


def overview_slice(config: dict) -> tuple[int, float]:
    budget = max(config['budgets'])
    theta = min(config['thresholds'], key=lambda value: (abs(value - .25), value))
    return budget, float(theta)


def runs_note(config: dict) -> str:
    return (f"{config['runs']} runs per pair with Random (mean over runs); pairs without Random played "
            f"{config['deterministic_runs']} times with the same outcome")


def heatmap(folder: Path, network: str, rows: list[dict], algorithms: list[str], k: int, theta: float,
            note: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    image = pair_heatmap(axis, rows, algorithms, 'mean_margin_b', 1, annotate=True)
    axis.set_title(f'{network}: ordered matchup margin\nk={k}, theta={theta:g}, sequential_a; '
                   'diagonal is self-play', fontsize=11)
    figure.colorbar(image, ax=axis, label='Mean margin (B - A) / n')
    figure.text(.5, .01, note, ha='center', fontsize=8)
    figure.tight_layout()
    for suffix in ('png', 'pdf'):
        figure.savefig(folder / f'{network}-matchup.{suffix}', dpi=160)
    plt.close(figure)
    write_csv(folder / f'{network}-matchup.csv', rows)


def draw(run_root: Path, output: Path) -> None:
    config = json.loads((run_root / 'config.json').read_text())
    k, theta = overview_slice(config)
    selected = {}
    for name in FILES['summaries']:
        for row in read_csv(run_root / 'analysis' / 'combined' / name):
            if int(row['k']) == k and float(row['theta']) == theta and row['allocation_order'] == 'sequential_a':
                selected.setdefault(row['graph_name'], []).append(row)
    output.mkdir(parents=True)
    for network in config['networks']:
        heatmap(output, network['name'], selected[network['name']], config['algorithms'], k, theta, runs_note(config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    draw(args.run_root, args.output)


if __name__ == '__main__':
    main()
