"""Single-condition example tables, read from runs/<batch>/results.csv.

The conditions are deterministic, so run 0 is used and run 1 is checked to be identical:
  priority-counterexample  ca-CondMat, Degree (A) vs Betweenness (B), k=10, theta=1/4, all sequences
  youtube-allocation       com-Youtube, Degree (A) vs Betweenness (B), k=20, theta=1/4, all sequences
  threshold-reversal       ca-CondMat, Degree (A) vs PageRank (B), k=20, snake, every threshold
                           from 1/8 to 5/16
Writes data/examples.csv.
"""
import pandas as pd

from common import OUT_DATA, SEQUENCES, THETA_FRACTION, batch_dir, output_path

RESULT_COLUMNS = ["graph", "nodes", "algorithm_a", "algorithm_b", "seed_count", "run_id", "draft_order", "theta",
                  "spread_a", "spread_b", "neutral", "rounds"]
REVERSAL_THETAS = [0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125]


def example_rows(network: str, k: int, theta: float, algorithm_a: str, algorithm_b: str, table: str,
                 sequences: list = SEQUENCES) -> list:
    results = pd.read_csv(f"{batch_dir(network, k, theta)}/results.csv", usecols=RESULT_COLUMNS)
    results = results[(results.algorithm_a == algorithm_a) & (results.algorithm_b == algorithm_b)
                      & results.draft_order.isin(sequences)]
    rows = []
    for sequence in sequences:
        runs = results[results.draft_order == sequence]
        assert len(runs) == 2 and runs.run_id.tolist() == [0, 1], (network, k, theta, sequence, len(runs))
        assert (runs[["spread_a", "spread_b", "neutral", "rounds"]].nunique() == 1).all(), "runs 0 and 1 differ"
        run = runs.iloc[0]
        rows.append(dict(table=table, network=network, k=k, theta=theta, sequence=sequence, algorithm_a=algorithm_a,
                         algorithm_b=algorithm_b, a_owned=int(run.spread_a), b_owned=int(run.spread_b),
                         diff=int(run.spread_b - run.spread_a), neutral=int(run.neutral),
                         neutral_pct=100 * run.neutral / run.nodes, rounds=int(run.rounds), nodes=int(run.nodes)))
    return rows


def main() -> None:
    rows = example_rows("ca-CondMat", 10, 0.25, "degree", "betweenness", "priority-counterexample")
    rows += example_rows("com-Youtube", 20, 0.25, "degree", "betweenness", "youtube-allocation")
    for theta in REVERSAL_THETAS:
        rows += example_rows("ca-CondMat", 20, theta, "degree", "pagerank", "threshold-reversal", ["snake"])
    examples = pd.DataFrame(rows)
    examples.to_csv(output_path(OUT_DATA, "examples.csv"), index=False)

    pd.set_option("display.width", 250)
    print(examples.drop(columns=["algorithm_a", "algorithm_b"]).round(2).to_string(index=False))
    reversal = examples[examples.table == "threshold-reversal"].set_index("theta")
    beyond_seeds = reversal.a_owned + reversal.b_owned - 2 * 20
    print("\nthreshold reversal: additional acquisitions (both players, seeds excluded):",
          {THETA_FRACTION[t]: int(v) for t, v in beyond_seeds.items()})


if __name__ == "__main__":
    main()
