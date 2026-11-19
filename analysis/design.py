"""Study configs: reading and checking a config, and the batches of matches it describes.

A config is a JSON file such as configs/full-study.json:

    {
      "networks": [{"name": "ca-GrQc", "path": "data/ca-GrQc.mtx"}, ...],
      "budgets": [20, 15, 10, 5],
      "thresholds": [0, 0.0625, ...],
      "runs": 100,
      "deterministic_runs": 2,
      ...
    }

Relative network paths are taken from the repository root. Every ordered pair of algorithms is
played (both roles and self-play) under every picking sequence. Pairs with Random are played
`runs` times; pairs without Random are deterministic and are played `deterministic_runs` times
(default: `runs`) so the analysis can check that every run gives the same outcome.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]

ALGORITHMS = ['random', 'degree', 'coverage_greedy', 'pagerank', 'betweenness']
SEQUENCES = ['sequential_a', 'sequential_b', 'alternating_a', 'snake']
NAME_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')

REQUIRED = {'networks', 'budgets', 'thresholds', 'runs'}
OPTIONAL = {'deterministic_runs', 'algorithms', 'sequences', 'base_seed', 'betweenness_threads',
            'jobs', 'description'}


@dataclass(frozen=True)
class Batch:
    """One engine run: every match of one network, budget and threshold."""
    id: str
    network: str
    budget: int
    threshold: float


def threshold_tag(theta: float) -> str:
    """Threshold as it appears in batch folder names: 0.25 -> '0p25', 0 -> '0', 1 -> '1'."""
    value = float(theta)
    text = str(int(value)) if value.is_integer() else repr(value)
    return text.replace('.', 'p')


def batch_id(network: str, budget: int, theta: float) -> str:
    return f'{network}-k{budget}-t{threshold_tag(theta)}'


def batch_dir(run_root: Path, network: str, budget: int, theta: float) -> Path:
    return Path(run_root) / 'runs' / batch_id(network, budget, theta)


def node_count(path: Path) -> int:
    """Number of nodes of a Matrix Market file, read from its size line."""
    with open(path) as stream:
        for line in stream:
            if not line.startswith('%'):
                rows, columns, _ = line.split()
                return max(int(rows), int(columns))
    raise ValueError(f'{path}: no size line')


def network_path(network: dict) -> Path:
    path = Path(network['path'])
    return path if path.is_absolute() else REPOSITORY / path


def load(path: Path) -> dict:
    """Reads a config, checks it, and fills in the defaults."""
    config = json.loads(Path(path).read_text())
    check(config)
    return with_defaults(config)


def with_defaults(config: dict) -> dict:
    full = dict(config)
    full.setdefault('deterministic_runs', config['runs'])
    full.setdefault('algorithms', ALGORITHMS)
    full.setdefault('sequences', SEQUENCES)
    full.setdefault('base_seed', 42)
    full.setdefault('betweenness_threads', 8)
    return full


def fail(message: str):
    raise ValueError('config: ' + message)


def check(config: dict) -> None:
    unknown = set(config) - REQUIRED - OPTIONAL
    if unknown:
        fail('unknown keys ' + ', '.join(sorted(unknown)))
    missing = REQUIRED - set(config)
    if missing:
        fail('missing keys ' + ', '.join(sorted(missing)))
    check_networks(config['networks'], config['budgets'])
    check_numbers(config)
    algorithms = config.get('algorithms', ALGORITHMS)
    if not algorithms or len(set(algorithms)) != len(algorithms) or not set(algorithms) <= set(ALGORITHMS):
        fail('algorithms must be distinct names among ' + ', '.join(ALGORITHMS))
    sequences = config.get('sequences', SEQUENCES)
    if len(set(sequences)) != len(sequences) or not set(sequences) <= set(SEQUENCES):
        fail('sequences must be distinct names among ' + ', '.join(SEQUENCES))
    if 'sequential_a' not in sequences:
        fail('sequences must include sequential_a, the reference of the sequence comparisons')


def check_networks(networks: list, budgets: list) -> None:
    if not isinstance(networks, list) or not networks:
        fail('networks must be a non-empty list')
    names = [network.get('name') for network in networks]
    if len(set(names)) != len(names):
        fail('network names must be distinct')
    for network in networks:
        name = network.get('name')
        if set(network) != {'name', 'path'} or not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            fail(f'each network needs a name (letters, digits, "_", ".", "-") and a path: {network}')
        path = network_path(network)
        if not path.is_file():
            fail(f'{name}: file {path} not found')
        if 2 * max(budgets) > node_count(path):
            fail(f'{name}: twice the largest budget exceeds the number of nodes')


def check_numbers(config: dict) -> None:
    def positive_integers(values) -> bool:
        return all(type(value) is int and value > 0 for value in values)

    budgets, thresholds = config['budgets'], config['thresholds']
    if not budgets or not positive_integers(budgets) or len(set(budgets)) != len(budgets):
        fail('budgets must be distinct positive integers')
    if (not thresholds or len({float(theta) for theta in thresholds}) != len(thresholds)
            or not all(type(theta) in (int, float) and 0 <= theta <= 1 for theta in thresholds)):
        fail('thresholds must be distinct numbers in [0, 1]')
    if type(config['runs']) is not int or config['runs'] < 2:
        fail('runs must be an integer of at least 2')
    deterministic = config.get('deterministic_runs', config['runs'])
    if type(deterministic) is not int or not 1 <= deterministic <= config['runs']:
        fail('deterministic_runs must be an integer from 1 to runs')
    for key in ('betweenness_threads', 'jobs'):
        if key in config and not positive_integers([config[key]]):
            fail(key + ' must be a positive integer')
    if 'base_seed' in config and (type(config['base_seed']) is not int or config['base_seed'] < 0):
        fail('base_seed must be a non-negative integer')


def batches(config: dict) -> list[Batch]:
    return [Batch(batch_id(network['name'], k, theta), network['name'], k, theta)
            for network in config['networks'] for k in config['budgets'] for theta in config['thresholds']]


def with_random(a: str, b: str) -> bool:
    return 'random' in (a, b)


def run_ids(config: dict, a: str, b: str) -> list[int]:
    """Run IDs of the pair (a, b): all runs with Random, the first deterministic_runs without."""
    return list(range(config['runs'] if with_random(a, b) else config['deterministic_runs']))


def pairs(config: dict) -> list[tuple[str, str]]:
    """Every ordered pair of algorithms, self-play included."""
    return [(a, b) for a in config['algorithms'] for b in config['algorithms']]


def planned_matches(config: dict) -> set[tuple]:
    """(algorithm A, algorithm B, sequence, run ID) of every match of one batch."""
    return {(a, b, sequence, run) for a, b in pairs(config) for sequence in config['sequences']
            for run in run_ids(config, a, b)}


def totals(config: dict) -> dict:
    """Conditions, matches and counted values of the whole study. A condition is a network,
    budget, threshold, sequence and ordered pair; without Random it counts as one value."""
    cells = len(config['networks']) * len(config['budgets']) * len(config['thresholds']) * len(config['sequences'])
    per_cell = [len(run_ids(config, a, b)) for a, b in pairs(config)]
    counted = [len(run_ids(config, a, b)) if with_random(a, b) else 1 for a, b in pairs(config)]
    return dict(conditions=cells * len(per_cell), executions=cells * sum(per_cell),
                condition_realizations=cells * sum(counted))


def design_part(config: dict) -> dict:
    """The config without the settings that do not change the results."""
    return {key: value for key, value in config.items() if key not in ('jobs', 'description')}
