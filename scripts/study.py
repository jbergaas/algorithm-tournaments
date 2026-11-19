#!/usr/bin/env python3
"""Run a study: Betweenness rankings, batches of matches, analysis, figures and a summary.

    python3 scripts/study.py run --config configs/full-study.json --out results/full-study [--jobs N]

Every step writes into its own folder under the output folder (first as <folder>.partial, renamed
when the step has finished). Running the same command again skips the finished steps, so an
interrupted study resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import design  # noqa: E402

REPOSITORY = design.REPOSITORY
DEFAULT_ENGINE = REPOSITORY / 'build/release/engine'


@dataclass
class Task:
    """One step of a study: a command that writes the folder `output`."""
    name: str
    command: list[str]
    output: Path
    log: Path
    slots: int = 1
    priority: tuple = ()
    needs: list[str] = field(default_factory=list)
    prepare: Optional[Callable[[Path], None]] = None   # writes inputs into the .partial folder

    @property
    def done(self) -> bool:
        return self.output.exists()


def python_command(script: str, *arguments) -> list[str]:
    return [sys.executable, '-B', str(REPOSITORY / 'analysis' / script), *map(str, arguments)]


def batch_config(config: dict, batch: design.Batch, run_root: Path) -> str:
    network = next(item for item in config['networks'] if item['name'] == batch.network)
    lines = [
        f'# Batch {batch.id}, written by scripts/study.py',
        f'graph = {design.network_path(network)}',
        f"algorithms = {', '.join(config['algorithms'])}",
        f'budget = {batch.budget}',
        f"runs = {config['runs']}",
        f"deterministic_runs = {config['deterministic_runs']}",
        f'theta = {batch.threshold}',
        f"sequences = {', '.join(config['sequences'])}",
        f"base_seed = {config['base_seed']}",
    ]
    if 'betweenness' in config['algorithms']:
        lines.append(f'betweenness_ranking = {run_root / "rankings" / batch.network / "fullranking.u32le"}')
    return '\n'.join(lines) + '\n'


def write_batch_config(config: dict, batch: design.Batch, run_root: Path, folder: Path) -> None:
    (folder / 'batch.cfg').write_text(batch_config(config, batch, run_root))


def ranking_tasks(config: dict, run_root: Path, engine: Path) -> list[Task]:
    if 'betweenness' not in config['algorithms']:
        return []
    tasks = []
    for network in config['networks']:
        path = design.network_path(network)
        output = run_root / 'rankings' / network['name']
        command = [str(engine), '--betweenness-ranking', str(path), '--output', str(output) + '.partial',
                   '--threads', str(config['betweenness_threads'])]
        tasks.append(Task(name='ranking ' + network['name'], command=command, output=output,
                          log=run_root / 'logs' / f"ranking-{network['name']}.log",
                          slots=config['betweenness_threads'], priority=(0, design.node_count(path))))
    return tasks


def batch_tasks(config: dict, run_root: Path, engine: Path) -> list[Task]:
    nodes = {network['name']: design.node_count(design.network_path(network)) for network in config['networks']}
    needs_ranking = 'betweenness' in config['algorithms']
    tasks = []
    for batch in design.batches(config):
        output = run_root / 'runs' / batch.id
        folder = Path(str(output) + '.partial')
        command = [str(engine), str(folder / 'batch.cfg'), '--output', str(folder / 'results.csv'),
                   '--trace', str(folder / 'rankings.csv')]
        tasks.append(Task(name='batch ' + batch.id, command=command, output=output,
                          log=run_root / 'logs' / f'batch-{batch.id}.log',
                          priority=(2, -nodes[batch.network]),
                          needs=['ranking ' + batch.network] if needs_ranking else [],
                          prepare=partial(write_batch_config, config, batch, run_root)))
    return tasks


def analysis_tasks(config: dict, run_root: Path) -> list[Task]:
    tasks = []
    batches = design.batches(config)
    for network in config['networks']:
        for k in config['budgets']:
            output = run_root / 'analysis' / network['name'] / f'k{k}'
            command = python_command('partition.py', '--run-root', run_root, '--network', network['name'],
                                     '--budget', k, '--output', str(output) + '.partial')
            needs = ['batch ' + batch.id for batch in batches if batch.network == network['name'] and batch.budget == k]
            tasks.append(Task(name=f"analysis {network['name']} k={k}", command=command, output=output,
                              log=run_root / 'logs' / f"analysis-{network['name']}-k{k}.log",
                              priority=(1,), needs=needs))
    return tasks


def final_tasks(run_root: Path, partitions: list[Task]) -> list[Task]:
    combined = run_root / 'analysis' / 'combined'

    def task(name: str, script: str, output: Path, needs: list[str]) -> Task:
        command = python_command(script, '--run-root', run_root, '--output', str(output) + '.partial')
        return Task(name=name, command=command, output=output, log=run_root / 'logs' / f'{name}.log',
                    priority=(3,), needs=needs)

    return [task('synthesis', 'synthesis.py', combined, [partition.name for partition in partitions]),
            task('figures', 'matchup_figures.py', run_root / 'figures', ['synthesis']),
            task('summary', 'summary.py', run_root / 'reports', ['synthesis'])]


def all_tasks(config: dict, run_root: Path, engine: Path) -> list[Task]:
    partitions = analysis_tasks(config, run_root)
    return (ranking_tasks(config, run_root, engine) + batch_tasks(config, run_root, engine)
            + partitions + final_tasks(run_root, partitions))


def execute(task: Task) -> float:
    """Runs one task into <output>.partial and renames that folder when the task has finished."""
    started = time.monotonic()
    folder = Path(str(task.output) + '.partial')
    if folder.exists():
        shutil.rmtree(folder)
    task.output.parent.mkdir(parents=True, exist_ok=True)
    if task.prepare is not None:
        folder.mkdir()
        task.prepare(folder)
    with open(task.log, 'w') as log:
        result = subprocess.run(task.command, stdout=log, stderr=subprocess.STDOUT, cwd=REPOSITORY)
    if result.returncode != 0:
        raise RuntimeError(f'{task.name} failed (exit code {result.returncode}); see {task.log}')
    folder.rename(task.output)
    return time.monotonic() - started


def run_tasks(tasks: list[Task], slots: int) -> None:
    """Runs the tasks in dependency order, at most `slots` slots at a time. A Betweenness ranking
    takes min(its threads, slots) slots but runs with all its threads.

    Among the ready tasks the one with the lowest priority value goes first. A Betweenness ranking that does not fit keeps the free slots
    for itself, unless another ranking is running (which frees enough slots when it ends).
    After a failure no new task starts.
    """
    finished = {task.name for task in tasks if task.done}
    waiting = sorted((task for task in tasks if not task.done), key=lambda task: task.priority)
    total, completed = len(waiting), 0
    running: dict[Future, Task] = {}
    free = slots
    failure = None
    with ThreadPoolExecutor(max_workers=slots) as pool:
        while waiting or running:
            if failure is None:
                for task in list(waiting):
                    if not all(name in finished for name in task.needs):
                        continue
                    needed = min(task.slots, slots)
                    if needed > free:
                        if task.slots > 1 and all(other.slots == 1 for other in running.values()):
                            break
                        continue
                    waiting.remove(task)
                    free -= needed
                    running[pool.submit(execute, task)] = task
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                free += min(task.slots, slots)
                try:
                    seconds = future.result()
                except Exception as error:
                    failure = failure or error
                    print(f'FAILED {task.name}: {error}', flush=True)
                    continue
                finished.add(task.name)
                completed += 1
                print(f'[{completed}/{total}] {task.name} ({seconds:.0f} s)', flush=True)
    if failure is not None:
        raise RuntimeError('the study stopped after a failure; fix it and run the same command again')
    if waiting:
        raise RuntimeError('some tasks could not start: ' + ', '.join(task.name for task in waiting[:5]))


def prepare_run_root(config: dict, run_root: Path) -> None:
    """Stores the config in the output folder, or checks that it matches the stored one."""
    stored = run_root / 'config.json'
    if stored.exists():
        if design.design_part(json.loads(stored.read_text())) != design.design_part(config):
            raise ValueError(f'{run_root} holds a study with a different config')
        return
    (run_root / 'logs').mkdir(parents=True, exist_ok=True)
    stored.write_text(json.dumps(config, indent=2) + '\n')


def run(args: argparse.Namespace) -> None:
    config = design.load(args.config)
    run_root = args.out.resolve()
    engine = args.engine.resolve()
    if not engine.is_file():
        raise ValueError(f'engine not found at {engine}; build it first (see README.md)')
    prepare_run_root(config, run_root)
    (run_root / 'logs').mkdir(exist_ok=True)
    jobs = args.jobs or config.get('jobs') or os.cpu_count()
    print(f"{len(config['networks'])} networks, {len(design.batches(config))} batches, "
          f"{design.totals(config)['executions']:,} matches; {jobs} jobs", flush=True)
    run_tasks(all_tasks(config, run_root, engine), jobs)
    print(f'Study complete: {run_root}', flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    run_parser = commands.add_parser('run', help='run (or resume) a study')
    run_parser.add_argument('--config', type=Path, required=True, help='study config (JSON)')
    run_parser.add_argument('--out', type=Path, required=True, help='output folder of the study')
    run_parser.add_argument('--jobs', type=int, help='parallel jobs (default: the config, else all cores)')
    run_parser.add_argument('--engine', type=Path, default=DEFAULT_ENGINE, help='engine executable')
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, OSError) as error:
        print('Error: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
