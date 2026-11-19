"""Diagnostic figures of one network and budget (PNG and PDF), drawn from the partition tables.

The threshold curves show sequential_a. The other figures show the smallest threshold. Every
cell, line or point is one ordered pair of algorithms.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAMES = {'random': 'Random', 'degree': 'Degree', 'coverage_greedy': 'Coverage Greedy',
         'pagerank': 'PageRank', 'betweenness': 'Betweenness'}


def save(figure, folder: Path, name: str) -> None:
    figure.tight_layout()
    for suffix in ('png', 'pdf'):
        figure.savefig(folder / f'{name}.{suffix}', dpi=145, bbox_inches='tight')
    plt.close(figure)


def pair_heatmap(axis, rows: list[dict], algorithms: list[str], field: str, limit: float,
                 annotate: bool = False, row_labels: bool = True):
    """One cell per ordered pair (A in the rows, B in the columns), coloured from red (A ahead,
    negative) to blue (B ahead, positive). Cells without a value stay grey."""
    index = {algorithm: position for position, algorithm in enumerate(algorithms)}
    matrix = np.full((len(algorithms), len(algorithms)), np.nan)
    for row in rows:
        matrix[index[row['algorithm_a']], index[row['algorithm_b']]] = float(row[field])
    colours = plt.get_cmap('RdBu').copy()
    colours.set_bad('#dddddd')
    image = axis.imshow(matrix, cmap=colours, vmin=-limit, vmax=limit)
    labels = [NAMES[a] for a in algorithms]
    axis.set_xticks(range(len(algorithms)), labels, rotation=35, ha='right')
    axis.set_xlabel('Algorithm of player B')
    if row_labels:
        axis.set_yticks(range(len(algorithms)), labels)
        axis.set_ylabel('Algorithm of player A')
    if annotate:
        for (a, b), value in np.ndenumerate(matrix):
            if np.isfinite(value):
                axis.text(b, a, f'{value:+.3f}', ha='center', va='center', fontsize=9)
    return image


def matchup_margins(folder, title, algorithms, summaries, theta):
    rows = [row for row in summaries if row['theta'] == theta and row['allocation_order'] == 'sequential_a']
    figure, axis = plt.subplots(figsize=(9, 7))
    image = pair_heatmap(axis, rows, algorithms, 'mean_margin_b', 1)
    axis.set_title(f'{title}, theta={theta:g}, sequential_a; diagonal is self-play')
    figure.colorbar(image, ax=axis, label='Mean margin (B - A) / n')
    save(figure, folder, 'ordered_matchup_margins')


def sequence_effects(folder, title, algorithms, contrasts, others, theta):
    rows = [row for row in contrasts if row['theta'] == theta]
    limit = max([abs(row['mean_delta_margin_b']) for row in rows] + [1e-12])
    figure, axes = plt.subplots(1, len(others), figsize=(5.7 * len(others), 6), sharey=True, squeeze=False)
    for axis, sequence in zip(axes[0], others):
        own = [row for row in rows if row['comparison_order'] == sequence]
        image = pair_heatmap(axis, own, algorithms, 'mean_delta_margin_b', limit, row_labels=axis is axes[0][0])
        axis.set_title(f'{sequence} minus sequential_a')
    figure.suptitle(f'{title}, theta={theta:g}: mean change of the margin (B - A) / n')
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=.7)
    save(figure, folder, 'allocation_effects_and_self_play')


def threshold_curves(folder, title, algorithms, overlap_rows, field, ylabel, name):
    rows = [row for row in overlap_rows if row['allocation_order'] == 'sequential_a']
    figure, axes = plt.subplots(1, len(algorithms), figsize=(3.8 * len(algorithms), 4.5), sharex=True,
                                sharey=True, squeeze=False)
    for axis, a in zip(axes[0], algorithms):
        for b in algorithms:
            own = sorted((row for row in rows if row['algorithm_a'] == a and row['algorithm_b'] == b),
                         key=lambda row: row['theta'])
            axis.plot([row['theta'] for row in own], [row[field] for row in own], '.-', label=NAMES[b],
                      linewidth=1)
        axis.set_title('A: ' + NAMES[a])
        axis.set_xlabel('Threshold theta')
        axis.set_ylim(-.02, 1.02)
        axis.grid(alpha=.2)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(title='B', fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    figure.suptitle(f'{title}, sequential_a')
    save(figure, folder, name)


def overlap_and_seed_changes(folder, title, overlap_rows, theta):
    rows = [row for row in overlap_rows if row['theta'] == theta and row['allocation_order'] == 'sequential_a']
    figure, axis = plt.subplots(figsize=(9, 6))
    for row in rows:
        point = (row['mean_top_k_overlap'], row['any_sequence_seed_pair_differs_proportion'])
        axis.scatter(*point, s=30)
        axis.annotate(f"{NAMES[row['algorithm_a']]} / {NAMES[row['algorithm_b']]}", point, fontsize=6)
    axis.set_xlabel('Mean overlap of the two top-k candidate lists (nodes)')
    axis.set_ylabel('Share of runs in which some sequence gives other seeds')
    axis.set_ylim(-.04, 1.04)
    axis.set_title(f'{title}, theta={theta:g}')
    save(figure, folder, 'prefix_overlap_and_seed_changes')


def random_intervals(folder, title, contrast_intervals, others, theta):
    rows = [row for row in contrast_intervals if row['theta'] == theta and row['random_involving']]
    if not rows:
        return
    pairs = sorted({(row['algorithm_a'], row['algorithm_b']) for row in rows})
    figure, axes = plt.subplots(1, len(others), figsize=(4.7 * len(others), 6), sharey=True, squeeze=False)
    for axis, sequence in zip(axes[0], others):
        by_pair = {(row['algorithm_a'], row['algorithm_b']): row for row in rows
                   if row['comparison_order'] == sequence}
        for position, pair in enumerate(pairs):
            row = by_pair[pair]
            axis.plot([row['interval_lower_95'], row['interval_upper_95']], [position, position], color='#456d82')
            axis.plot(row['mean'], position, 'o', color='#182c39', markersize=4)
        axis.axvline(0, color='.7', linewidth=.7)
        axis.set_title(f'{sequence} - sequential_a')
        axis.set_xlabel('Change of the margin')
    axes[0][0].set_yticks(range(len(pairs)), [f'{NAMES[a]} (A) / {NAMES[b]} (B)' for a, b in pairs])
    axes[0][0].invert_yaxis()
    figure.suptitle(f'{title}, theta={theta:g}: means and 95% bootstrap intervals over the Random runs')
    save(figure, folder, 'random_allocation_uncertainty')


def draw(folder: Path, network: str, k: int, config: dict, summaries, contrasts, overlap_rows,
         contrast_intervals) -> None:
    folder.mkdir()
    algorithms = config['algorithms']
    others = [sequence for sequence in config['sequences'] if sequence != 'sequential_a']
    theta = float(min(config['thresholds']))
    title = f'{network}, k={k}'
    matchup_margins(folder, title, algorithms, summaries, theta)
    if others:
        sequence_effects(folder, title, algorithms, contrasts, others, theta)
        random_intervals(folder, title, contrast_intervals, others, theta)
    threshold_curves(folder, title, algorithms, overlap_rows, 'mean_neutral_fraction', 'Neutral fraction',
                     'neutral_vs_threshold')
    threshold_curves(folder, title, algorithms, overlap_rows, 'mean_acquired_beyond_seeds_fraction',
                     'Share of nodes acquired beyond the 2k seeds', 'acquisitions_vs_threshold')
    overlap_and_seed_changes(folder, title, overlap_rows, theta)
