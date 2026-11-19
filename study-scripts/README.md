# Study scripts

Scripts that made the input networks, turn a finished run into the tables and figures of the
thesis, and compute the other numbers in it. The study pipeline (`scripts/study.py`) does not use
them. They need the packages in `requirements.txt`.

## input-prep/

How the networks in `data/` were made is described in `data/README.md`.

| Script | What it does |
|---|---|
| `snap_to_mtx.py INPUT.txt.gz OUT_DIR NAME SOURCE_URL` | Converts a SNAP edge list into a simple undirected graph without isolated nodes (`canonical.mtx`), and keeps the original entries, the SNAP ID of each node and a `conversion.json`. Used for ca-HepTh, soc-Epinions1 and ego-Facebook. |
| `generate_networks.py OUT_DIR [--n 10000] [--instances 3] [--only SETTING ...]` | Makes the Erdos-Renyi, Chung-Lu, Watts-Strogatz and coauthorship networks. The seed of each network is the first 8 hex digits of the SHA-256 of its name, so with the versions in `requirements.txt` it reproduces the files in `data/` byte for byte. `--n 50000 --only er-d8 cl-g2.5` makes the larger networks. |
| `erdos_converter.py --input Erdos02.net --output-dir NEW_DIR` | Rebuilds Erdos02 from the original Pajek file. The result is identical to `data/Erdos02.mtx`. |

## tables/

These scripts make the result tables from a finished run. Run them from this folder in this order,
since later scripts read the files written by earlier ones:

```
python3 s01_network_stats.py
python3 s02_medians.py
python3 s03_overlap.py
python3 s04_sequences.py
python3 s05_topk_share.py
python3 s06_algorithms.py
python3 s07a_random_runs.py
python3 s07_elo.py
python3 s08_examples.py
python3 s09_tables.py
python3 ../figures/adoption_threshold.py out out/figures/adoption_threshold.pdf
python3 s12_text_numbers.py | tee out/data/text_numbers.out
```

The whole sequence takes about a minute. The scripts write CSV files to `out/data/` and LaTeX
tables to `out/tables/`, and print the numbers the thesis quotes; `s12_text_numbers` only prints.
`s09_tables` also writes a few tables that are not in the final thesis, among them Table 4.3 in an
older layout.

| Environment variable | Default | Meaning |
|---|---|---|
| `STUDY_RUN_ROOT` | `<repo>/results/full-study` | The run to read |
| `STUDY_TABLES_OUT` | `tables/out/` | Where to write (use it in place of `out` above) |

The graphs are read from `data/` of this repository. `average_path_lengths.csv` holds the average
shortest-path length of each network, estimated by breadth-first search from 500 random sources
(200 above 100,000 nodes, seed 20261105, sampling error about 0.02-0.03).

## figures/

Each script writes the figure as PDF and a PNG preview, and all but `generated_families.py` also
write the plotted values as CSV.

| Script | Figure |
|---|---|
| `generated_families.py OUT` | Figure 3.3: small examples of the random, uneven and ring networks, written to `OUT.pdf` and `OUT.png`. Reads no study data. |
| `adoption_threshold.py TABLES_OUT OUT.pdf` | Figure 4.1: median additional acquisition per real network and threshold at k=20 under snake, with the collapse marked. Run it after `tables/s02_medians.py`. |
| `block_imbalance.py outcome_changes.csv OUT.pdf` | Figure 4.2: block imbalance per real network against how often swapping A-first and B-first block changes the winner. Input from `tables/s04_sequences.py`. |
| `first_pick.py role_adv_by_graph.csv OUT.pdf` | Figure 4.3: median first-pick advantage per real network and sequence, before and after the collapse. Input from `analyses/first_pick/advantage.py`. |
| `predictors.py same_games.csv.gz OUT.pdf` | Figure 4.4: how often the player ahead on the distance race, first-round capture and access wins, by threshold. Input from `analyses/predictors/same_games.py`. |
| `netstyle.py` | Colours, markers and network order shared by the figures. |

## analyses/

The scripts behind the other numbers in the thesis: where the spread collapses and what moves it,
the one-neighbour test, the first-pick advantage, the stability of the ratings, the winner
predictors, what knowing more is worth and the sensitivity to the diffusion model. There is also a
Python version of the engine's allocation and diffusion that replays the stored matches.
`analyses/README.md` gives the order to run them in and, for each number in the thesis, the script
and output line it comes from. These scripts also need numba.
