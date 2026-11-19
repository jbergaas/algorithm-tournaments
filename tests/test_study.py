"""End-to-end check of scripts/study.py on the ten-node test graph.

Needs the engine built in build/release (see README.md). Run from the repository root:

    python3 -m unittest discover -s tests
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from statistics import fmean

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = {
    'networks': [{'name': 'toy', 'path': 'tests/data/test.mtx'}],
    'budgets': [2],
    'thresholds': [0.25, 0.5],
    'runs': 5,
    'deterministic_runs': 2,
    'betweenness_threads': 2,
    'jobs': 2,
}


def read(path: Path) -> list[dict]:
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


class StudyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        (root / 'config.json').write_text(json.dumps(CONFIG))
        cls.run_root = root / 'run'
        cls.first = cls.study(root / 'config.json')
        cls.second = cls.study(root / 'config.json')

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    @classmethod
    def study(cls, config: Path) -> subprocess.CompletedProcess:
        command = [sys.executable, '-B', str(REPOSITORY / 'scripts/study.py'), 'run',
                   '--config', str(config), '--out', str(cls.run_root)]
        return subprocess.run(command, capture_output=True, text=True)

    def test_runs_and_resumes(self):
        self.assertEqual(self.first.returncode, 0, self.first.stderr)
        self.assertIn('[7/7]', self.first.stdout)
        self.assertEqual(self.second.returncode, 0, self.second.stderr)
        self.assertNotIn('[1/', self.second.stdout, 'a finished step was run again')

    def test_every_match_is_counted(self):
        # 9 pairs with Random x 5 runs + 16 pairs without x 2 runs, 4 sequences, 2 thresholds.
        coverage = read(self.run_root / 'analysis/combined/network_coverage.csv')
        self.assertEqual(coverage, [{'graph_name': 'toy', 'conditions': '200', 'executions': '616',
                                     'condition_realizations': '488'}])

    def test_summaries_match_the_raw_results(self):
        results = read(self.run_root / 'runs/toy-k2-t0p25/results.csv')
        summaries = read(self.run_root / 'analysis/toy/k2/condition_summaries_cross_play.csv')
        for summary in summaries:
            rows = [row for row in results if float(row['theta']) == float(summary['theta'])
                    and row['draft_order'] == summary['allocation_order']
                    and (row['algorithm_a'], row['algorithm_b']) == (summary['algorithm_a'], summary['algorithm_b'])]
            if not rows:
                continue
            margins = [(int(row['spread_b']) - int(row['spread_a'])) / int(row['nodes']) for row in rows]
            if summary['random_involving'] == 'True':
                self.assertEqual(len(rows), 5)
                self.assertEqual(float(summary['mean_margin_b']), fmean(margins))
            else:
                self.assertEqual(len(rows), 2)
                self.assertEqual(len(set(margins)), 1)
                self.assertEqual(float(summary['mean_margin_b']), margins[0])
                self.assertEqual(summary['scientific_realization_count'], '1')

    def test_intervals_contain_the_mean(self):
        for row in read(self.run_root / 'analysis/combined/condition_uncertainty.csv'):
            if row['random_involving'] == 'True':
                self.assertLessEqual(float(row['interval_lower_95']), float(row['mean']))
                self.assertLessEqual(float(row['mean']), float(row['interval_upper_95']))
            else:
                self.assertEqual(row['interval_lower_95'], '')

    def test_betweenness_plays_its_stored_ranking(self):
        ranking = (self.run_root / 'rankings/toy/fullranking.u32le').read_bytes()
        first_four = [int.from_bytes(ranking[4 * i:4 * i + 4], 'little') for i in range(4)]
        for row in read(self.run_root / 'runs/toy-k2-t0p5/rankings.csv'):
            if row['algorithm_a'] == 'betweenness':
                self.assertEqual(json.loads(row['ranking_a']), first_four)


if __name__ == '__main__':
    unittest.main()
