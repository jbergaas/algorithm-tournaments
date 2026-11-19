"""Paths, study constants and loaders shared by the result-table and figure scripts.

The tables cover the 11 real networks and the 14 thresholds of the study. The generated networks
are not loaded here. The graphs are read from data/ of this repository. The run root and the
output folder can be set through environment variables (see study-scripts/README.md). Output
folders are created when a script first writes to them (output_path).
"""
import os
from fractions import Fraction

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))
GRAPH_DIR = f"{REPO}/data"
AVG_PATH_CSV = f"{SCRIPT_DIR}/average_path_lengths.csv"
RUN_ROOT = os.environ.get("STUDY_RUN_ROOT", f"{REPO}/results/full-study")
COMBINED_DIR = f"{RUN_ROOT}/analysis/combined"
RUNS_DIR = f"{RUN_ROOT}/runs"
OUTPUT_DIR = os.environ.get("STUDY_TABLES_OUT", f"{SCRIPT_DIR}/out")
OUT_DATA = f"{OUTPUT_DIR}/data"
OUT_TABLES = f"{OUTPUT_DIR}/tables"

# Table row order.
NETWORKS = ["Erdos02", "ca-GrQc", "wiki-Vote", "ca-HepPh", "ca-CondMat", "email-Enron", "com-DBLP",
            "com-Youtube", "ca-HepTh", "soc-Epinions1", "ego-Facebook"]
NODES = {"ca-GrQc": 5241, "Erdos02": 6927, "wiki-Vote": 7115, "ca-HepPh": 12006, "ca-CondMat": 23133,
         "email-Enron": 36692, "com-DBLP": 317080, "com-Youtube": 1134890,
         "ca-HepTh": 9875, "soc-Epinions1": 75879, "ego-Facebook": 4039}
THETAS = [0.0, 0.0625, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.4375,
          0.5, 0.75]
POSITIVE_THETAS = [t for t in THETAS if t > 0]
THETA_FRACTION = {t: str(Fraction(t).limit_denominator(64)) for t in THETAS}   # 0.15625 -> "5/32"
BUDGETS = (5, 10, 15, 20)

ALGORITHMS = ["degree", "coverage_greedy", "pagerank", "betweenness", "random"]
STRUCTURAL = ["degree", "coverage_greedy", "pagerank", "betweenness"]
LETTER = {"degree": "D", "coverage_greedy": "C", "pagerank": "P", "betweenness": "B", "random": "R"}
SEQUENCES = ["sequential_a", "sequential_b", "alternating_a", "snake"]
ELO_SEED = 20261104   # seed of the reported Elo ratings (s07_elo also rates with seeds 1 and 2)

SUMMARY_COLUMNS = ["graph_name", "k", "theta", "allocation_order", "algorithm_a", "algorithm_b",
                   "random_involving", "is_self_play", "scientific_realization_count",
                   "mean_a_fraction", "mean_b_fraction", "mean_neutral_fraction", "mean_margin_b",
                   "mean_acquisition_rounds", "mean_ranking_overlap",
                   "strict_a_win_proportion", "strict_b_win_proportion", "draw_proportion"]
CONDITION_KEY = ["graph_name", "k", "theta", "allocation_order", "algorithm_a", "algorithm_b"]


def output_path(folder: str, name: str) -> str:
    """Path of `name` in an output folder (OUT_DATA or OUT_TABLES), creating the folder if needed."""
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def batch_dir(network: str, k: int, theta: float) -> str:
    """Folder runs/<network>-k<k>-t<tag> of one batch.

    The tag is theta with "." written as "p" (0.25 -> 0p25, 0.0625 -> 0p0625); a whole-number
    theta is written as that number (0 -> 0, 1 -> 1).
    """
    tag = str(int(theta)) if theta == int(theta) else repr(float(theta)).replace(".", "p")
    return f"{RUNS_DIR}/{network}-k{k}-t{tag}"


def is_deterministic_cross_play(row: dict) -> bool:
    """Run 0 of a condition between two different structural algorithms (a row of results.csv or
    rankings.csv read with csv.DictReader)."""
    return (row["run_id"] == "0" and row["algorithm_a"] in STRUCTURAL and row["algorithm_b"] in STRUCTURAL
            and row["algorithm_a"] != row["algorithm_b"])


def load_combined(name: str, columns: list) -> pd.DataFrame:
    """Rows of analysis/combined/<name>.csv for the study's networks and thresholds.

    Thresholds are rounded to 6 decimals and the boolean columns parsed.
    """
    parts = []
    reader = pd.read_csv(os.path.join(COMBINED_DIR, f"{name}.csv"), usecols=columns, low_memory=False,
                         chunksize=200_000)
    for chunk in reader:
        parts.append(chunk[chunk["graph_name"].isin(NETWORKS)])
    table = pd.concat(parts, ignore_index=True)
    table["theta"] = table["theta"].astype(float).round(6)
    table = table[table["theta"].isin(THETAS)].reset_index(drop=True)
    for column in ("random_involving", "is_self_play"):
        if column in table:
            table[column] = table[column].astype(str).str.lower().eq("true")
    return table


def cross_play_summaries() -> pd.DataFrame:
    return load_combined("condition_summaries_cross_play", SUMMARY_COLUMNS)


def self_play_summaries() -> pd.DataFrame:
    return load_combined("self_play_diagnostics", SUMMARY_COLUMNS)


def overlap_conditions() -> pd.DataFrame:
    return load_combined("overlap_conditions_cross_play",
                         CONDITION_KEY + ["random_involving", "scientific_realization_count",
                                          "top_k_overlap_eligible_count"])


def condition_uncertainty() -> pd.DataFrame:
    return load_combined("condition_uncertainty",
                         CONDITION_KEY + ["random_involving", "mean", "interval_lower_95", "interval_upper_95",
                                          "bootstrap_block_id"])


def with_additional_adoption(summaries: pd.DataFrame) -> pd.DataFrame:
    """Add q, the share of the network acquired beyond the 2k seeds."""
    nodes = summaries["graph_name"].map(NODES)
    return summaries.assign(q=summaries["mean_a_fraction"] + summaries["mean_b_fraction"]
                            - 2 * summaries["k"] / nodes)


def own_margins(conditions: pd.DataFrame) -> pd.DataFrame:
    """One row per condition and algorithm in it: alg, opp (opponent), role and own (own mean margin)."""
    as_a = conditions.assign(alg=conditions.algorithm_a, opp=conditions.algorithm_b, role="A",
                             own=-conditions.mean_margin_b)
    as_b = conditions.assign(alg=conditions.algorithm_b, opp=conditions.algorithm_a, role="B",
                             own=conditions.mean_margin_b)
    return pd.concat([as_a, as_b], ignore_index=True)


def threshold_regimes(summaries: pd.DataFrame) -> pd.DataFrame:
    """Regime of every (network, k, theta).

    R1: theta <= 1/16. R2: theta > 1/16 and median adoption >= 1/2. R3: theta > 1/16 and median
    adoption < 1/2. Adoption is 1 - mean neutral fraction (seeds included); the median runs over
    the 80 cross-play conditions (20 ordered pairs including Random, four sequences).
    """
    cross = summaries[~summaries.is_self_play]
    adoption = 1 - cross.mean_neutral_fraction
    regimes = adoption.groupby([cross.graph_name, cross.k, cross.theta]).agg(["median", "size"])
    regimes = regimes.rename(columns={"median": "median_adoption", "size": "conditions"}).reset_index()
    assert (regimes.conditions == 80).all(), regimes.conditions.unique()
    regimes["regime"] = np.where(regimes.theta <= 1 / 16, "R1",
                                 np.where(regimes.median_adoption >= 0.5, "R2", "R3"))
    return regimes


def read_graph(name: str):
    """Undirected simple graph of data/<name>.mtx as CSR arrays.

    Returns (n, start, neighbours, edge_u, edge_v): the neighbours of node i are
    neighbours[start[i]:start[i + 1]], and every edge appears once as edge_u < edge_v.
    Self-loops and duplicate edges are dropped.
    """
    with open(f"{GRAPH_DIR}/{name}.mtx") as f:
        f.readline()                      # MatrixMarket banner
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        n = int(line.split()[0])
        edges = np.loadtxt(f, dtype=np.int64, usecols=(0, 1), ndmin=2) - 1
    edges = edges[edges[:, 0] != edges[:, 1]]
    low = np.minimum(edges[:, 0], edges[:, 1])
    high = np.maximum(edges[:, 0], edges[:, 1])
    pair_ids = np.unique(low * n + high)
    edge_u, edge_v = pair_ids // n, pair_ids % n
    source = np.concatenate([edge_u, edge_v])
    target = np.concatenate([edge_v, edge_u])
    order = np.argsort(source, kind="stable")
    start = np.searchsorted(source[order], np.arange(n + 1))
    return n, start, target[order], edge_u, edge_v
