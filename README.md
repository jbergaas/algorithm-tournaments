# Algorithm tournaments for competitive influence maximization

Code and data for the master's thesis *Algorithm Tournaments for Competitive Influence
Maximization* (Jonas Bergaas, University of Bergen, 2026). A C++ engine plays tournaments between
seed-selection algorithms, and Python scripts run a whole study and analyse it.

## The game

Two players, A and B, each place k seeds on the same network, using one of five algorithms:
Random, Degree, Coverage Greedy, PageRank or Betweenness. Each algorithm lists 2k candidate nodes,
and a picking sequence hands out the seeds, skipping nodes the other player already holds. The
four sequences are A-first block (`sequential_a`), B-first block (`sequential_b`), strict
alternation (`alternating_a`) and snake.

The seeds then spread under a competitive linear threshold rule with one threshold θ for all
nodes. In each round, every neutral node joins the player that owns more of its neighbours, if
that player owns at least a share θ of them. Owned nodes never change. The spread stops when a
round changes nothing, and the outcome is the number of nodes each player owns.

Every ordered pair of algorithms, self-play included, plays under every sequence. Pairs with
Random are played many times with different seeds; pairs without Random are deterministic and are
played twice to check that both runs agree. The included study (`configs/full-study.json`) has 56
networks, budgets 5, 10, 15 and 20, 14 thresholds and 100 runs per pair with Random: 11,691,008
matches in 3,136 batches.

## Layout

| Path | Contents |
|---|---|
| `include/engine/`, `src/` | The engine |
| `tools/` | C++ tests |
| `tests/` | Python end-to-end test and the test graphs |
| `preprocessing/` | `isolate_filter`, which removes nodes without edges |
| `scripts/study.py` | Runs a study |
| `analysis/` | Analyses a run |
| `configs/` | The full study and a short smoke test |
| `data/` | The 56 networks, with sources and licences (`data/README.md`) |
| `study-scripts/` | The scripts behind the tables, figures and other numbers of the thesis, with a table from each number to its script (`study-scripts/README.md`) |
| `third_party/` | fast_matrix_market 1.7.7, the Matrix Market reader |

## Requirements

- A C++20 compiler with OpenMP, and CMake 3.20 or newer.
- Python 3.10 with the packages in `requirements.txt`.
- For the full study, many cores: on 16 cores with 62 GiB of memory it took about 14 hours, most
  of it the Betweenness of com-Youtube. The output takes about 10 GiB.

The study was run on Linux (Linux Mint 21.3, GCC 11.4, CPython 3.10.9); other systems have not
been tested. The build uses `-march=native`, so a binary is tied to the CPU it was built on.

## Build and test

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel
ctest --test-dir build/release --output-on-failure
python3 -m unittest discover -s tests
```

The C++ tests check the engine on small graphs with known answers and on six real networks. The
Python test runs a whole study on a ten-node graph.

## Run a study

```bash
python3 scripts/study.py run --config configs/smoke.json --out results/smoke          # a few seconds
python3 scripts/study.py run --config configs/full-study.json --out results/full-study --jobs 30
```

The script computes the Betweenness ranking of each network once (`rankings/`), plays one batch
per network, budget and threshold (`runs/`), and analyses the results per network and budget
(`analysis/`) and for all networks together (`analysis/combined/`, `figures/`,
`reports/summary.md`). The analysis also checks that every batch holds the planned matches and
that the deterministic runs agree. Running the same command again continues an interrupted study.
Logs are in `<out>/logs/`.

A config looks like this:

```json
{
  "networks": [{"name": "ca-GrQc", "path": "data/ca-GrQc.mtx"}, ...],
  "budgets": [20, 15, 10, 5],
  "thresholds": [0, 0.0625, 0.125, ...],
  "runs": 100,
  "deterministic_runs": 2,
  "algorithms": ["random", "degree", "coverage_greedy", "pagerank", "betweenness"],
  "sequences": ["sequential_a", "sequential_b", "alternating_a", "snake"],
  "base_seed": 42,
  "betweenness_threads": 16,
  "jobs": 30
}
```

`networks`, `budgets`, `thresholds` and `runs` are required; the rest have defaults. Paths are
relative to the repository root. `sequential_a` must be among the sequences, since the others are
compared with it, and twice the largest budget must not exceed the number of nodes of any network.

### Running the engine directly

```bash
build/release/engine --betweenness-ranking data/ca-GrQc.mtx --threads 16 --output rankings/ca-GrQc
build/release/engine my.cfg --output results.csv --trace rankings.csv
```

The first command writes the Betweenness scores and ranking of a network. The scores count each
pair of nodes in both directions, twice the usual value, which leaves the ranking the same. The
second plays one batch, every match of one network, budget and threshold, from a `key = value`
file:

```
graph = data/ca-GrQc.mtx
algorithms = random, degree, coverage_greedy, pagerank, betweenness
budget = 20                     # seeds per player
runs = 100                      # runs of each pair with Random
deterministic_runs = 2          # runs of each pair without Random
theta = 0.25
sequences = sequential_a, sequential_b, alternating_a, snake
base_seed = 42
betweenness_ranking = rankings/ca-GrQc/fullranking.u32le   # only when betweenness is listed
```

`results.csv` gets one row per match: the algorithms, the sequence, the run and the nodes each
player owns. Its `edges` column counts every edge in both directions. `rankings.csv` gets both
candidate lists and both seed sets of every match.

## Reproducibility

- Random's seed in a match depends only on the base seed, the network, the player's algorithm, the
  budget, the run and the player. In one run, Random therefore picks the same candidates against
  every opponent, under every sequence and at every threshold.
- Betweenness is computed in parallel but adds up the contributions in a fixed order, so the
  scores are the same for any number of threads. Ties are broken by node ID.
- Random's candidates match the study's exactly only with GCC's standard library, because the C++
  standard does not fix how `std::shuffle` uses its random generator.

## Licence

MIT, see `LICENSE`. `third_party/fast_matrix_market/` keeps its own licence. The networks keep the
terms of their providers (see `data/README.md`).
