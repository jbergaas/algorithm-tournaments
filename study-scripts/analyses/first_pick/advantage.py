"""First-pick advantage per network and picking sequence, before and after the collapse, and which
sequence is fairest.

Writes out/role_adv_by_graph.csv and out/advantage.txt with the numbers the text quotes. The CSV has
one row per network and sequence. Its prefixes all_, S_ and N_ cover all settings, those before the
collapse and those after it. Per prefix: number of pair-settings, median, mean and mean absolute
advantage in percent of the nodes, share favouring and disfavouring the first picker, and share in
which the first picker wins both role assignments, loses both, or the same algorithm wins both.
spread_share is the share of the settings before the collapse.
Usage: python3 first_pick/advantage.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402
from common import role_advantage  # noqa: E402

OUT = s.out_dir("first_pick")
BLOCK_LEADERS = ["com-Youtube", "ego-Facebook", "soc-Epinions1", "Erdos02", "email-Enron"]
COLLABORATION = ["ca-GrQc", "ca-HepTh", "ca-HepPh", "ca-CondMat", "com-DBLP"]
DOMINANT_HUBS = ["Erdos02", "soc-Epinions1", "com-Youtube"]


def summarise(x: pd.DataFrame) -> pd.Series:
    return pd.Series(dict(n=len(x), med=100 * x.fp_adv.median(), mean=100 * x.fp_adv.mean(),
                          mean_abs=100 * x.fp_adv.abs().mean(), pos=100 * (x.fp_adv > 1e-12).mean(),
                          neg=100 * (x.fp_adv < -1e-12).mean(), fp_both=100 * x.fp_both.mean(),
                          sp_both=100 * x.sp_both.mean(), alg_both=100 * x.alg_both.mean()))


def by_graph(ra: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for label, sub in (("all", ra), ("S", ra[ra.spread]), ("N", ra[~ra.spread])):
        parts.append(sub.groupby(["graph_name", "allocation_order"]).apply(summarise).add_prefix(f"{label}_"))
    out = pd.concat(parts, axis=1)
    out["spread_share"] = 100 * out.S_n / out.all_n
    return out.reset_index()


def text_numbers(ra: pd.DataFrame, bg: pd.DataFrame) -> list[str]:
    before = bg.pivot(index="graph_name", columns="allocation_order", values="S_med")
    after = bg.pivot(index="graph_name", columns="allocation_order", values="N_med")
    real = before.index.isin(s.REAL)
    coauthorship = before.index.str.startswith("collab-")
    pooled = ra[ra.spread].groupby("allocation_order")
    lines = ["Median first-pick advantage before the collapse, A-first block (% of the nodes):",
             "  " + ", ".join(f"{g} {before.loc[g, 'sequential_a']:.1f}" for g in BLOCK_LEADERS),
             f"  arXiv and DBLP networks: {before.loc[COLLABORATION, 'sequential_a'].min():.1f} to "
             f"{before.loc[COLLABORATION, 'sequential_a'].max():.1f}",
             "Median after the collapse, largest over the sequences: generated "
             f"{after[~real].max().max():.2f}, real {after[real].max().max():.2f}",
             "Pooled over the 56 networks before the collapse:"]
    for seq in ("sequential_a", "alternating_a", "snake"):
        x = pooled.get_group(seq)
        lines.append(f"  {seq:13s} median advantage {100 * x.fp_adv.median():.2f}, first picker wins both role "
                     f"assignments in {100 * x.fp_both.mean():.2f}% of the pair-settings")
    size = before[["alternating_a", "snake"]].abs()
    alternation_smaller = int((size.alternating_a < size.snake).sum())
    snake_smaller = int((size.snake < size.alternating_a).sum())
    pairs = [f"{g} {before.loc[g, 'alternating_a']:.1f}/{before.loc[g, 'snake']:.1f}" for g in s.REAL]
    lines += [f"Smaller |median| before the collapse: strict alternation on {alternation_smaller}"
              f" of {len(size)} networks, snake on {snake_smaller}",
              "Real networks, median before the collapse (strict alternation / snake):",
              "  " + ", ".join(pairs),
              "Snake before the collapse: " + ", ".join(f"{g} {before.loc[g, 'snake']:.1f}" for g in DOMINANT_HUBS)
              + f"; generated coauthorship graphs {before[coauthorship].snake.min():.1f} to "
              f"{before[coauthorship].snake.max():.1f}"]
    return lines


def main() -> None:
    ra = role_advantage(s.cross_play())
    bg = by_graph(ra)
    bg.to_csv(OUT / "role_adv_by_graph.csv", index=False)
    text = "\n".join(text_numbers(ra, bg))
    (OUT / "advantage.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
