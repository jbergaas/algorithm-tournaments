# Additional analyses

The scripts behind the numbers in the thesis that do not come from the result tables or figures
(`../tables`, `../figures`). They read a finished run and the network files, and write only to their
own `out/` folders. Besides the packages in `requirements.txt` they need numba, which compiles the
simulators.

`shared.py` holds the paths, the study constants, the network loader, the readers of the stored
run, and a numba version of the engine's seed allocation and diffusion. It reads the networks the
same way as the engine.

| Environment variable | Default | Meaning |
|---|---|---|
| `STUDY_RUN_ROOT` | `<repo>/results/full-study` | The run to read |
| `STUDY_DATA` | `<repo>/data` | The network files |
| `STUDY_ANALYSES_OUT` | this folder | Where the `out/` folders and `out/cache/` go |
| `STUDY_TABLES_OUT` | `../tables/out` | Output of `../tables`, which `ratings/interleaved.py` checks against |

Parsed networks, run tables and numba's compiled code are cached in `out/cache/`. Delete it after
changing the network files or when reading another run with the same folder name.

## Order and commands

Run from this folder, in this order; each script reads only the run and the outputs of the scripts
above it. The number after a script is the number of processes (default 4). The times are for 4
processes on the 16-core machine of the study, with empty caches; scripts without a time take a few
seconds.

```bash
python3 collapse/observed.py               # collapse intervals; needed by watts, predictors
python3 collapse/degree_duel.py 4          # 12 s
python3 collapse/rewire.py 4               # 7 min (com-Youtube dominates)
python3 collapse/summary.py
python3 collapse/generated_table.py        # 21 s
python3 watts/one_neighbour.py
python3 watts/summary.py
python3 mechanisms/race.py 4               # 47 s
python3 mechanisms/race_summary.py
python3 mechanisms/degree_vs_random_hepph.py
python3 mechanisms/top_degree_group.py
python3 mechanisms/low_degree_neighbours.py 4
python3 capture/matches.py 4               # 2 min (6 min with 2 processes)
python3 predictors/same_games.py           # needs collapse, mechanisms/race.py and capture
python3 predictors/race_interleaved.py
python3 first_pick/advantage.py
python3 first_pick/self_play.py
python3 first_pick/reach_share.py
python3 first_pick/shared_top_k.py
python3 ratings/variants.py                # 51 s
python3 ratings/interleaved.py             # run ../tables first for the check against elo_all.csv
python3 ratings/generated_winners.py
python3 knowledge/value_of_knowledge.py
python3 diffusion_sensitivity/simulate.py 8   # 53 min with 8 processes
python3 diffusion_sensitivity/race.py 4
python3 diffusion_sensitivity/analyse.py
python3 checks/pagerank.py
```

`../figures/first_pick.py` and `../figures/predictors.py` draw Figures 4.3 and 4.4 from
`first_pick/out/role_adv_by_graph.csv` and `predictors/out/same_games.csv.gz`.

## The scripts

Each script writes to `out/` in its own folder. The index at the end says which numbers in the
thesis each one produces.

| Script | What it computes |
|---|---|
| **collapse/** | |
| `observed.py` | The collapse interval of every network and budget: the first sampled threshold above 0 where the median adoption over the 80 cross-algorithm conditions is below one half, and the threshold before it. |
| `degree_duel.py` | At every threshold, one Degree-vs-Degree match and 2k random seeds spreading alone. |
| `rewire.py` | The deterministic matches replayed on a copy of each network, rewired at random with every degree kept. |
| `summary.py` | The collapse intervals from the scripts above side by side, and the numbers quoted about them. |
| `generated_table.py` | Table 4.1, the generated networks at k = 20. It computes every cell; the thesis prints the random-seed column only for the uneven graphs and the rewired column only for the clustered ring graphs. |
| **watts/** | |
| `one_neighbour.py` | Per network and threshold, the nodes that need only one owned neighbour and the largest connected cluster they form. |
| `summary.py` | The test based on Watts's condition, and Table 4.2. |
| **mechanisms/** | |
| `race.py` | The distance race of every deterministic match: how many more nodes are nearer to one player's seeds than to the other's. |
| `race_summary.py` | How often the race leader wins, and the race leader of each network. |
| `degree_vs_random_hepph.py` | Which algorithms lose any condition to Random on the real networks, and why Degree does on ca-HepPh. |
| `top_degree_group.py` | Degree's top 10 on each real network and the edges among them. |
| `low_degree_neighbours.py` | The neighbours of each algorithm's top k, each counted as one over its degree; PageRank against Degree. |
| **capture/** | |
| `matches.py` | Replays all 150,528 deterministic matches between two different structural algorithms and checks them against the stored results. At k = 10 and 20 it also records each player's first-round capture and access. |
| **predictors/** | |
| `games.py` | Module: the games both scripts use (k = 10 and 20, the real networks and the first instance of each generated setting). |
| `same_games.py` | The three predictors (distance race, first-round capture, access) on the same games; input of Figure 4.4. |
| `race_interleaved.py` | The distance race under strict alternation and snake. |
| **first_pick/** | |
| `common.py` | Module: which settings lie before the collapse, and the first-pick advantage of every pair. |
| `advantage.py` | The first-pick advantage per network and sequence, before and after the collapse; input of Figure 4.3. |
| `self_play.py` | The median self-play lead per network and sequence. |
| `reach_share.py` | The first picker's share of the adopted nodes against its share of the seeds' reach. |
| `shared_top_k.py` | How many of their top k two algorithms share. |
| **ratings/** | |
| `games.py` | Module: the threshold regimes and the games between the structural algorithms. |
| `elo.py` | Module: Elo averaged over random game orders, and the mean score. |
| `variants.py` | The 260 variants of the ratings. |
| `interleaved.py` | The ratings without the block sequences. |
| `generated_winners.py` | Elo per generated network and regime. |
| **knowledge/** | |
| `value_of_knowledge.py` | Table 4.5: how often a player wins that picks the best structural algorithm for what it knows. |
| **diffusion_sensitivity/** | |
| `variants.py` | Module: the changed diffusion models. |
| `simulate.py` | Replays the stored matches under each change, and checks the unchanged model against the stored results. |
| `race.py` | The distance race of every replayed match. |
| `analyse.py` | Ties, collapse shifts, regime winners, first-pick advantage and predictors for each change. |
| **checks/** | |
| `pagerank.py` | PageRank recomputed with scipy and compared with the engine's ranking. |

`diffusion_sensitivity/` replays the 64 deterministic matches of run 0 at budgets 10 and 20 and all
14 thresholds, on the 11 real networks and er-d8-i0, cl-g2.5-i0, ws-p0.01-i0 and collab-s50-i0,
with one assumption of the model changed at a time: every threshold 1% lower or higher, a threshold
per node spread around theta, three threshold levels, or ties settled by a fair coin. Each random
variant is sampled 20 times, seeded from 20261105 and the network's position in
`variants.NETWORKS`. The replay took 53 minutes with 8 processes, most of it on com-Youtube and
com-DBLP. If `simulate.py` is interrupted, delete `out/units/` and start again, so that every unit
is checked.

## Where each number comes from

Paths are relative to this folder. Numbers marked "tables" come from `../tables`.

| Thesis place | Numbers | Script | Output |
|---|---|---|---|
| Methods, Validation (`sec:methods-validation`) | the re-implementation reproduces all 150,528 deterministic matches between the structural algorithms | `capture/matches.py` | `capture/out/validation.txt` |
| Methods, Validation (`sec:methods-validation`), "every stored match checked" | includes the 26,880 base matches of the sensitivity check (the number is not printed in the thesis) | `diffusion_sensitivity/simulate.py` | `diffusion_sensitivity/out/validation.csv` |
| Results 4.1.1 | one Degree-vs-Degree match finds the collapse within one interval on ten of the eleven real networks at k = 20 and in 205 of 224 cases; misses by more than one on ca-HepPh, two clustered ring graphs and the three collab-s200 graphs | `collapse/summary.py` | `collapse/out/summary.txt`, first block |
| Results 4.1.2 | the interval moves to higher thresholds from k = 5 to k = 20 on 43 of 56 networks, to lower on none | `collapse/summary.py` (also printed by `observed.py`) | `collapse/out/summary.txt`, second block |
| Results 4.1.2 | per-budget intervals: email-Enron and com-DBLP do not move; wiki-Vote at k = 5 between 1/16 and 1/8 | `collapse/observed.py` | `collapse/out/observed_collapse.csv` |
| Table 4.1 (`tab:results-generated-collapse`) | average degree, clustering, degree share, collapse, random seeds, rewired, and the footnote on the instances | `collapse/generated_table.py` | `collapse/out/generated_table.txt` (per setting), `generated_table.csv` (per network) |
| Results 4.1.3 | random graphs collapse earlier as the average degree rises; uneven graphs three intervals after er-d8 and ws-p1.0; degree share 6 to 14% against under 1%; size check: same interval for er-d8, one earlier for the uneven graphs, 9.9% to 5.1% | `collapse/generated_table.py` | `collapse/out/generated_table.txt` |
| Results 4.1.3 | with 40 random seeds alone, eight of the nine uneven graphs collapse where the random graphs do | `collapse/summary.py` | `collapse/out/summary.txt`, last block |
| Results 4.1.3, Discussion 5.1 | the same matches on the rewired copies of the clustered ring graphs collapse 4, 4 and 1 intervals earlier (ws-p0.1) and 5, 5 and 5 (ws-p0.01); on the real networks the rewiring moves the collapse up to three intervals earlier, most on ego-Facebook and com-Youtube | `collapse/summary.py` | `collapse/out/summary.txt`, third block |
| Results 4.1.4 | largest one-neighbour cluster below 3% at theta = 1/8 on every real network | `watts/summary.py` | `watts/out/summary.txt`, block 2 |
| Results 4.1.4 | all 224 cases last at least up to the 10% threshold, 154 more than one step beyond; cutoffs 1, 5, 25, 50%: all 224, 133 to 155 | `watts/summary.py` | `watts/out/summary.txt`, block 1 |
| Results 4.1.4, Table 4.2 (`tab:results-one-neighbour`) | gap, one-neighbour share and largest cluster per group at k = 20; at most 0.1% before the collapse although 9 to 88% need one owned neighbour | `watts/summary.py` | `watts/out/summary.txt`, block 4 |
| Results 4.1.4, Discussion 5.1 | adoption without seeds exceeds the one-neighbour share on nine of the eleven real networks, by 12 to 45% of the network; about equal on ca-GrQc, lower on Erdos02 | `watts/summary.py` | `watts/out/summary.txt`, block 3 |
| Results 4.2.1 | at k = 20 every pair of different structural algorithms shares at least one top candidate on all 56 networks; on Erdos02, email-Enron, soc-Epinions1 and com-Youtube two algorithms share 70 to 79% of their top 20 (mean over the six pairs), on ca-GrQc and ca-HepPh about 30% | `first_pick/shared_top_k.py` | `first_pick/out/shared_top_k.txt` |
| Results 4.2.1, 4.3.1, Discussion 5.4 | the ten highest-degree nodes form a clique on ca-GrQc and ca-HepPh only; on ca-HepPh their degrees add up to 4,500 with 964 distinct neighbours; ca-CondMat 18 of 45 edges | `mechanisms/top_degree_group.py` | `mechanisms/out/top_degree_group.txt` |
| Results 4.2.1 | com-Youtube, k = 20, theta = 1/4, Degree (A) against Betweenness (B): the two rankings share 13 of their top 20; under a block sequence the first picker gets all 13 and wins; under strict alternation and snake they split 7 to 6 and 90.8% of the network stays neutral | tables: `../tables/s08_examples.py` for the outcomes; the shared candidates and seeds are in the batch's `rankings.csv` | `../tables/out/data/examples.csv` (`youtube-allocation`); `runs/com-Youtube-k20-t0p25/rankings.csv` |
| Results 4.2.2 | median first-pick advantage under A-first block before the collapse: 84, 78, 71, 65, 52% on com-Youtube, ego-Facebook, soc-Epinions1, Erdos02, email-Enron; 4 to 34% on the arXiv and DBLP networks; first picker wins both role assignments in 74%; after the collapse below 2% (generated) and at most 5.6% (real) | `first_pick/advantage.py` | `first_pick/out/advantage.txt`; `role_adv_by_graph.csv` columns `S_med`, `N_med` |
| Results 4.2.2, Fig. 4.3 (`fig:results-first-pick`) | medians per network and sequence | `first_pick/advantage.py`, then `../figures/first_pick.py` | `first_pick/out/role_adv_by_graph.csv` |
| Results 4.2.2 | a seed-reach share of 55 to 60% gives about 70 to 79% of the adopted nodes before the collapse on the random, uneven and coauthorship networks and 63% on the real networks; on the ring graphs a larger reach share goes with a smaller adopted share; about the same share after the collapse | `first_pick/reach_share.py` | `first_pick/out/reach_share.txt`, column (0.55, 0.6] |
| Results 4.2.3 | pooled median before the collapse 50.2% (block), 4.6% (strict alternation), 1.2% (snake) | `first_pick/advantage.py` | `first_pick/out/advantage.txt`, "Pooled" |
| Results 4.2.3 | Erdos02 self-play lead 57.82% to 4.14% | `first_pick/self_play.py` | `first_pick/out/self_play.txt` |
| Results 4.2.3, Discussion 5.2 | strict alternation smaller on 11 of 56 networks, snake on 35; ego-Facebook 2.1 against 14.8, Erdos02 25.5 against 30.6; snake 26, 21, 44% on Erdos02, soc-Epinions1, com-Youtube and 6 to 17% on the coauthorship graphs | `first_pick/advantage.py` | `first_pick/out/advantage.txt`, last three blocks |
| Results 4.3.1 | in all 260 variants PageRank has the highest mean in the spreading regime and over all thresholds; Degree first on six networks in 230 and on five in the rest; the same counts without the block sequences | `ratings/variants.py` | `ratings/out/variants_summary.txt` |
| Results 4.3.1 | without the block sequences every network keeps its winner per regime; PageRank 77 points ahead instead of 28; over all thresholds ca-GrQc PageRank to Betweenness and soc-Epinions1 Coverage Greedy to PageRank | `ratings/interleaved.py` | `ratings/out/interleaved_summary.txt` |
| Results 4.3.2 | 38 of 44 combinations positive against Random in all 416 conditions; five of the six exceptions lose by at most 2% | `mechanisms/degree_vs_random_hepph.py` | `mechanisms/out/degree_vs_random_hepph.txt`, "Pairs ... that lose" |
| Results 4.3.2 | Degree's mean margin -0.463 and 38 wins of 100 runs (Random as A, k = 10, 3/16, snake) | tables: `../tables/s06_algorithms.py` | `../tables/out/data/random_degree.csv` |
| Results 4.3.2 | the top 20 of ca-HepPh one group with 380 to 491 neighbours; half of the nodes have at most five neighbours; Degree alone reaches less than 1% except ten seeds at 5/32; with 15 or more seeds Degree wins every condition, by three quarters on average | `mechanisms/degree_vs_random_hepph.py` | `mechanisms/out/degree_vs_random_hepph.txt` |
| Results 4.3.3 | winners on the generated networks per regime; PageRank most frequent spreading-regime winner in 7 of 13 settings, Degree in 2; ws-p0.01: Betweenness on two instances, Degree on one, Coverage Greedy last on all three; 9 to 17 of Betweenness's 20 seeds next to an own seed; games above the collapse are near-ties | `ratings/generated_winners.py` | `ratings/out/generated_winners.txt` |
| Results 4.3.3, Discussion 5.7 | instances of a setting differ by up to about 600 rating points | `ratings/generated_winners.py` | `ratings/out/generated_winners.txt`, "largest difference" |
| Results 4.4 | the race leader wins 92% (theta = 0) and 87% (1/16) of the decided games | `mechanisms/race_summary.py` | `mechanisms/out/race_by_match.csv` (`agree_decisive`), `race_summary.txt` |
| Results 4.4, Discussion 5.4 | Betweenness leads the race on most networks | `mechanisms/race_summary.py` | `mechanisms/out/race_summary.txt` (34 of 56, 9 of 11 real) |
| Results 4.4 | under strict alternation and snake the race leader still wins 85% and 77% | `predictors/race_interleaved.py` | `predictors/out/race_interleaved.txt` (84.7%, 76.6%) |
| Results 4.4, Discussion 5.4, Appendix A.2 | capture 80%, race 74%, access 89%; block: capture 91%, access 95%; interleaved: capture 68%, race 60%, access 84%; own top k 61% | `predictors/same_games.py` | `predictors/out/same_games.txt` |
| Fig. 4.4 (`fig:results-predictors`) | the points; 552 to 1,130 games per point | `predictors/same_games.py`, then `../figures/predictors.py` | `predictors/out/same_games.csv.gz`; per-threshold lines of `same_games.txt` |
| Results 4.4, Discussion 5.4 | with every neighbour counted as one over its degree, PageRank's top nodes score at least as high as Degree's on every network at budgets of 10 and more, and on all but one at a budget of 5 | `mechanisms/low_degree_neighbours.py` | `mechanisms/out/low_degree_neighbours.txt` |
| Table 4.5 (`tab:results-knowledge`) and its caption | all rows; fewer than 1% draws, at most 0.2 points | `knowledge/value_of_knowledge.py` | `knowledge/out/knowledge.txt` (columns `score`, `wins`, `draws`), `levels.csv` |
| Results 4.5 | gain mostly from wiki-Vote, ca-HepPh and email-Enron; PageRank best at every budget on five other networks; the best against all four is a best reply in 98%; the opponent, sequence and role known before the threshold gain 3.1 points (level `opponent, sequence and role, no threshold`); the second picker under a block sequence wins about a third | `knowledge/value_of_knowledge.py` | `knowledge/out/knowledge.txt` |
| Introduction 1.5, Discussion 5.5, Conclusion | 66%, 72%, 60%; less than two points more | `knowledge/value_of_knowledge.py` | `knowledge/out/knowledge.txt` |
| Appendix A.2, Results 4.6 | ties: 70% of the matches with a node neutral only because of a tie, 0.38% of the network on average, at most 3.8%; random tie resolution moves no collapse, changes no regime winner on the real networks, and the predictors by at most two points | `diffusion_sensitivity/analyse.py` | `diffusion_sensitivity/out/summary.txt`, ties, collapse table (row `tie`), regime winners, predictor tables |
| Appendix A.2, Results 4.6 | collapse: same interval in 83% of the cases at w = 0.10, almost all exceptions one earlier; most stay at w = 0.25; +1% moves 7 of 30 one interval earlier; later by about one interval at w = 0.5 and two with three levels | `diffusion_sensitivity/analyse.py` | `diffusion_sensitivity/out/summary.txt`, collapse table |
| Appendix A.2 | first-pick advantage under A-first block 49.1 to 50.5 before the collapse and 0.4 to 0.8 after it | `diffusion_sensitivity/analyse.py` | `diffusion_sensitivity/out/summary.txt`, "Median first-pick advantage" |
| Appendix A.2, Results 4.6 | low-threshold winner kept in every sample on the real networks, in at least 87% on the generated; spreading-regime winners unchanged after averaging, single samples at least 90% except soc-Epinions1 at w = 0.25 (62%); cl-g2.5 changes at the two widest spreads, ws-p0.01 under +1%; post-collapse winners often change | `diffusion_sensitivity/analyse.py` | `diffusion_sensitivity/out/summary.txt`, regime winners (`winners.csv`) |
| Appendix A.2 | race leader wins 84 to 89% of the decided games at theta <= 1/16 in every variant; capture of the own top k about 65% up to w = 0.25, base 66%, 61% at w = 0.5, 58% with three levels | `diffusion_sensitivity/race.py`, `analyse.py` | `diffusion_sensitivity/out/summary.txt`, predictor tables (`predictors.csv`) |
| Appendix B (numerical checks) | the re-implemented PageRank reproduces the engine's top 40 on all 56 networks; tolerance reached by iteration 100 on all but ca-GrQc (104); top 40 unchanged at 500 iterations | `checks/pagerank.py` | `checks/out/pagerank.txt` |
