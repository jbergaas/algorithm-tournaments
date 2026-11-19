"""Write the result tables as LaTeX from the CSVs in data/.

Writes tables/<label>.tex for every table. Captions and header text are fixed; the counts in them
come from the data. Run it after s01-s08.
"""
import os
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from common import (ALGORITHMS, BUDGETS, ELO_SEED, LETTER, NETWORKS, OUT_DATA, OUT_TABLES, POSITIVE_THETAS,
                    THETA_FRACTION, THETAS, output_path)

ADOPTION_THETAS = [0.125, 0.1875, 0.25, 0.3125, 0.375, 0.75]
APPENDIX_THETAS = [0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.5, 0.75]
SEQUENCE_NAME = {"sequential_a": "A-first block", "sequential_b": "B-first block",
                 "alternating_a": "Strict alternation", "snake": "Snake"}
REGIMES = ("R1", "R2", "R3", "ALL")


# Number formatting
def half_up(x: float, digits: int = 0) -> str:
    """Round half away from zero, on the value printed with 9 decimals (avoids binary ties)."""
    d = Decimal(f"{x:.9f}").quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{d:.{digits}f}"


def percent(x: float) -> str:
    """One decimal; values below 0.05 are shown as <0.1."""
    return "$<0.1$" if x < 0.05 else f"{x:.1f}"


def signed(x: float, fmt: str) -> str:
    """Negative numbers in math mode, so the minus sign is typeset correctly."""
    text = format(x, fmt)
    return f"${text}$" if text.startswith("-") else text


def thousands(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def theta_math(theta: float) -> str:
    return f"${THETA_FRACTION[theta]}$"


# LaTeX assembly
def table_row(cells: list) -> str:
    return "        " + " & ".join(cells) + r" \\"


def float_table(label: str, caption: str, spec: str, header: str, body: list, short: str = "",
                after_label: str = "", before_caption: str = "", size: str = "small") -> str:
    """A table float with a booktabs tabular."""
    caption_line = f"\\caption[{short}]{{{caption}}}" if short else f"\\caption{{{caption}}}"
    lines = ["\\begin{table}[htbp]", "    \\centering", f"    \\{size}"]
    if before_caption:
        lines.append(f"    {before_caption}")
    lines += [f"    {caption_line}", f"    \\label{{{label}}}"]
    if after_label:
        lines.append(f"    {after_label}")
    lines += [f"    \\begin{{tabular}}{{{spec}}}", "        \\toprule", header, "        \\midrule", *body,
              "        \\bottomrule", "    \\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def write_table(label: str, text: str) -> None:
    with open(output_path(OUT_TABLES, f"{label}.tex"), "w") as f:
        f.write(text.rstrip() + "\n")


def read_data(name: str, **options) -> pd.DataFrame:
    return pd.read_csv(f"{OUT_DATA}/{name}", **options)


# Diffusion
def adoption(stats: pd.DataFrame, snake: pd.DataFrame) -> None:
    columns = len(ADOPTION_THETAS)
    caption = ("Median additional acquisition as a percentage of all nodes, with $k=20$ and \\seq{snake}, at six "
               "thresholds. Each entry is the median of the 20 cross-algorithm condition means. Initial seeds are "
               "excluded. The first three columns give the average degree and the share of nodes in the largest "
               "component, both from \\cref{tab:methods-network-properties}, and the share of nodes with exactly "
               "one neighbour, in percent.")
    spec = "@{}lrrr@{\\hspace{10pt}}" + "r" * columns + "@{}"
    header = ("        & \\multicolumn{3}{c}{Network} & \\multicolumn{" + str(columns)
              + "}{c}{Threshold \\(\\theta\\)} \\\\\n"
              "        \\cmidrule(lr){2-4}\\cmidrule(l){5-" + str(4 + columns) + "}\n"
              "        Network & Avg.\\ deg. & Largest & Deg.\\ 1 & "
              + " & ".join(theta_math(t) for t in ADOPTION_THETAS) + " \\\\")
    body = [table_row([network, f"{stats.loc[network, 'avg_deg']:.1f}", f"{100 * stats.loc[network, 'largest']:.1f}",
                       f"{stats.loc[network, 'deg1_pct']:.1f}"]
                      + [percent(snake.loc[(network, 20, t), "q_pct"]) for t in ADOPTION_THETAS])
            for network in NETWORKS]
    write_table("tab:results-adoption",
                float_table("tab:results-adoption", caption, spec, header, body,
                            after_label="\\setlength{\\tabcolsep}{4pt}"))


def rounds(stats: pd.DataFrame, medians: pd.DataFrame) -> None:
    snake20 = medians[(medians.allocation_order == "snake") & (medians.k == 20)]
    rounds_by_theta = snake20.pivot(index="graph_name", columns="theta", values="rounds")
    caption = ("Median acquisition rounds\n"
               "    with \\(k=20\\) and \\seq{snake}, over the 20 cross-algorithm condition\n"
               "    means, rounded to whole rounds, and the largest distance from the 40\n"
               "    nodes of highest degree to any other node in the same component.")
    header = ("        Network & Distance & Rounds at \\(\\theta=0\\) & Most rounds (at \\(\\theta\\)) & "
              "Rounds at \\(\\theta=3/4\\) \\\\")
    body = []
    for network in NETWORKS:
        per_theta = rounds_by_theta.loc[network, THETAS]
        peak_theta = per_theta.idxmax()
        body.append(table_row([network, str(int(stats.loc[network, "dist_from_top40"])), half_up(per_theta[0.0]),
                               f"{half_up(per_theta.max())} (\\({THETA_FRACTION[peak_theta]}\\))",
                               half_up(per_theta[0.75])]))
    write_table("tab:results-rounds", float_table("tab:results-rounds", caption, "@{}lrrrr@{}", header, body,
                                                  short="Acquisition rounds and distances"))


def budget(stats: pd.DataFrame, snake: pd.DataFrame) -> None:
    caption = ("Median additional acquisition (\\%) at two thresholds under \\seq{snake}, at the smallest and largest "
               "budget. Each value is the median of the 20 cross-algorithm condition means. The first two columns "
               "give the share of the network (\\%) that lies next to the \\(2k\\) nodes of highest degree, from "
               "which Degree picks its candidates.")
    header = ("        & \\multicolumn{2}{c}{Reach of top \\(2k\\)} & \\multicolumn{2}{c}{$\\theta=1/4$} & "
              "\\multicolumn{2}{c}{$\\theta=5/16$} \\\\\n"
              "        \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\n"
              "        Network & $k=5$ & $k=20$ & $k=5$ & $k=20$ & $k=5$ & $k=20$ \\\\")
    body = [table_row([network, f"{stats.loc[network, 'reach_top2k_k5_pct']:.1f}",
                       f"{stats.loc[network, 'reach_top2k_k20_pct']:.1f}"]
                      + [percent(snake.loc[(network, k, t), "q_pct"]) for t in (0.25, 0.3125) for k in (5, 20)])
            for network in NETWORKS]
    write_table("tab:results-budget",
                float_table("tab:results-budget", caption, "@{}lrr@{\\hspace{10pt}}rrrr@{}", header, body,
                            after_label="\\setlength{\\tabcolsep}{5pt}"))


def appendix(snake: pd.DataFrame) -> None:
    """Median additional acquisition for every budget, as a longtable."""
    columns = len(APPENDIX_THETAS) + 2
    head = "Network & $k$ & " + " & ".join(theta_math(t) for t in APPENDIX_THETAS) + " \\\\"
    lines = ["\\begingroup", "\\small", "\\setlength{\\tabcolsep}{2.4pt}",
             f"\\begin{{longtable}}{{@{{}}l{'r' * (len(APPENDIX_THETAS) + 1)}@{{}}}}",
             "\\caption[Additional acquisition across all budgets]{Median additional acquisition (\\%) under snake. Each "
             "entry summarizes 20 ordered cross-algorithm means, excluding the initial seeds. Values below 0.05\\% "
             "are shown as $<0.1$.}",
             "\\label{tab:appendix-adoption-budgets}\\\\", "\\toprule", head, "\\midrule", "\\endfirsthead",
             f"\\multicolumn{{{columns}}}{{l}}{{\\tablename\\ \\thetable\\ -- continued}}\\\\", "\\toprule",
             head, "\\midrule", "\\endhead", "\\midrule",
             f"\\multicolumn{{{columns}}}{{r}}{{Continued on the next page}}\\\\", "\\endfoot", "\\bottomrule",
             "\\endlastfoot"]
    for i, network in enumerate(NETWORKS):
        if i:
            lines.append("\\addlinespace[0.3em]")
        for k in BUDGETS:
            cells = [network, str(k)] + [percent(snake.loc[(network, k, t), "q_pct"]) for t in APPENDIX_THETAS]
            lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\end{longtable}", "\\endgroup"]
    write_table("tab:appendix-adoption-budgets", "\n".join(lines))


# Picking sequences
def similarity() -> None:
    shares = read_data("similarity.csv")
    at_k20 = shares[shares.k == 20].set_index("network")
    at_k5 = shares[shares.k == 5].set_index("network")
    caption = ("Share (\\%) of the top \\(k\\) candidates that two deterministic algorithms have in common, for "
               "the six pairs at \\(k=20\\), and the mean over the six pairs at \\(k=5\\) and \\(k=20\\). D: Degree; "
               "C: Coverage Greedy; P: PageRank; B: Betweenness.")
    header = ("        & \\multicolumn{6}{c}{Pairs at \\(k=20\\)} & \\multicolumn{2}{c}{Mean} \\\\\n"
              "        \\cmidrule(lr){2-7}\\cmidrule(l){8-9}\n"
              "        Network & D--C & D--P & D--B & C--P & C--B & P--B & \\(k=5\\) & \\(k=20\\) \\\\")
    pairs = ["D-C", "D-P", "D-B", "C-P", "C-B", "P-B"]
    body = [table_row([network] + [f"{at_k20.loc[network, pair]:.0f}" for pair in pairs]
                      + [half_up(at_k5.loc[network, "mean"]), half_up(at_k20.loc[network, "mean"])])
            for network in NETWORKS]
    write_table("tab:results-similarity",
                float_table("tab:results-similarity", caption, "@{}lrrrrrr@{\\hspace{10pt}}rr@{}", header, body,
                            short="How much the deterministic algorithms agree",
                            after_label="\\setlength{\\tabcolsep}{5pt}"))


def overlap() -> None:
    counts = read_data("random_overlap.csv", index_col=0)
    caption = ("Cases in which Random's top \\(k\\) candidates share at least one node with the opponent's "
               "(\\(h_k>0\\)), out of 800 per network and budget (eight ordered pairs with 100 runs each). Exp.\\ is "
               "the number expected by chance (hypergeometric model).")
    header = ("        & \\multicolumn{2}{c}{$k=5$} & \\multicolumn{2}{c}{$k=20$} \\\\\n"
              "        \\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n        Network & Obs. & Exp. & Obs. & Exp. \\\\")
    body = [table_row([network, str(counts.loc[network, "obs_k5"]), f"{counts.loc[network, 'exp_k5']:.1f}",
                       str(counts.loc[network, "obs_k20"]), f"{counts.loc[network, 'exp_k20']:.1f}"])
            for network in NETWORKS]
    write_table("tab:results-overlap", float_table("tab:results-overlap", caption, "@{}lrrrr@{}", header, body,
                                                   short="Shared top candidates with Random"))


def outcome_changes() -> None:
    changes = read_data("outcome_changes.csv", index_col=0)
    settings = int(changes.settings.iloc[0])
    caption = ("Changes in the win/draw outcome relative to \\seq{A-first block}, out of "
               f"{settings} deterministic cross-algorithm settings per network: 12 ordered pairs, four budgets, and "
               f"{len(POSITIVE_THETAS)} positive thresholds. Draw-to-win and win-to-draw changes are included. "
               "Swing is the median change in B's margin, in percentage points, between \\seq{A-first block} and "
               "\\seq{B-first block} in the settings where the outcome changes. The last two columns give the mean "
               "share of top 20 candidates that the deterministic algorithms have in common "
               "(\\cref{tab:results-similarity}) and the coefficient of variation of the degrees "
               "(\\cref{tab:methods-network-properties}).")
    header = (f"        & \\multicolumn{{3}}{{c}}{{Outcome changes (of {settings})}} & & & \\\\\n"
              "        \\cmidrule(lr){2-4}\n"
              "        Network & B-first & Altern. & Snake & Swing & Shared (\\%) & Deg.\\ CV \\\\")
    body = [table_row([network, str(changes.loc[network, "B_first"]), str(changes.loc[network, "altern"]),
                       str(changes.loc[network, "snake"]), f"{changes.loc[network, 'swing_pp']:.1f}",
                       f"{changes.loc[network, 'shared_k20']:.0f}", f"{changes.loc[network, 'deg_cv']:.2f}"])
            for network in NETWORKS]
    write_table("tab:results-outcome-changes",
                float_table("tab:results-outcome-changes", caption, "@{}lrrrr@{\\hspace{10pt}}rr@{}", header, body,
                            after_label="\\setlength{\\tabcolsep}{5pt}"))


def priority_counterexample(examples: pd.DataFrame) -> None:
    label = "tab:results-priority-counterexample"
    caption = ("Giving Betweenness the first block reverses its win into a loss on ca-CondMat. Degree plays A, "
               "Betweenness plays B, $k=10$, and $\\theta=1/4$. Both rankings are fixed. Counts include the seeds.")
    header = "        Picking sequence & A-owned & B-owned & $u_B-u_A$ & Rounds \\\\"
    body = [table_row([SEQUENCE_NAME[r.sequence], thousands(r.a_owned), thousands(r.b_owned),
                       f"${'+' if r['diff'] > 0 else '-'}{thousands(abs(r['diff']))}$", f"{r.rounds}"])
            for _, r in examples[examples.table == "priority-counterexample"].iterrows()]
    write_table(label, float_table(label, caption, "@{}lrrrr@{}", header, body))


def youtube_allocation(examples: pd.DataFrame) -> None:
    label = "tab:results-youtube-allocation"
    caption = ("Degree as A and Betweenness as B on com-Youtube, \\(k=20\\), \\(\\theta=1/4\\). Both rankings are "
               "fixed across sequences. Counts include the initial seeds.")
    header = "        Picking sequence & A-owned & B-owned & Neutral (\\%) & Rounds \\\\"
    body = [table_row([SEQUENCE_NAME[r.sequence], thousands(r.a_owned), thousands(r.b_owned), f"{r.neutral_pct:.1f}",
                       f"{r.rounds}"])
            for _, r in examples[examples.table == "youtube-allocation"].iterrows()]
    write_table(label, float_table(label, caption, "@{}lrrrr@{}", header, body,
                                   short="Allocation changes both the winner and spread"))


def threshold_reversal(examples: pd.DataFrame) -> None:
    label = "tab:results-threshold-reversal"
    rows = examples[examples.table == "threshold-reversal"].copy()
    rows["theta"] = rows.theta.round(6)
    caption = ("Threshold-dependent reversal on ca-CondMat: Degree as A, PageRank as B, $k=20$, with \\seq{snake} "
               "allocation. The rankings and allocated seeds are fixed across rows. Owned-node counts include 20 seeds "
               "per player.")
    header = "        $\\theta$ & A-owned & B-owned & Neutral (\\%) & Rounds \\\\"
    body = [table_row([theta_math(r.theta), thousands(r.a_owned), thousands(r.b_owned), f"{r.neutral_pct:.1f}",
                       f"{r.rounds}"]) for _, r in rows.iterrows()]
    write_table(label, float_table(label, caption, "@{}rrrrr@{}", header, body))


def topk_share() -> None:
    shares = read_data("topk_share.csv", index_col=0)
    caption = ("How often the player that receives the larger share of its own top \\(k\\) candidates wins, for "
               f"pairs of different deterministic algorithms at the {len(POSITIVE_THETAS)} positive thresholds, all "
               "budgets, and all four sequences. Shared is the mean share of the top \\(k\\) that the two rankings "
               "have in common. Settings in which both players receive the same share are left out.")
    header = "        Network & Shared (\\%) & Settings & Larger share wins (\\%) \\\\"
    body = [table_row([network, half_up(shares.loc[network, "shared_pct"]), f"{int(shares.loc[network, 'settings']):,}",
                       f"{shares.loc[network, 'larger_wins_pct']:.1f}"]) for network in NETWORKS]
    write_table("tab:results-topk-share", float_table("tab:results-topk-share", caption, "@{}lrrr@{}", header, body,
                                                      short="Receiving the top candidates"))


def selfplay() -> None:
    balance = read_data("selfplay_balance.csv", index_col=0)
    settings = int(balance.settings.iloc[0])
    caption = ("Deterministic self-play: the median size of the lead, \\(|m_B|\\), in percent of the network, under "
               f"each sequence, and the number of the {settings} settings (four algorithms, four budgets, "
               f"{len(POSITIVE_THETAS)} positive thresholds) in which the lead under \\seq{{strict alternation}} or "
               "\\seq{snake} is smaller than, equal to, or larger than under \\seq{A-first block}.")
    header = ("        & \\multicolumn{3}{c}{Median lead (\\%)} & \\multicolumn{3}{c}{Alternation vs block} & "
              "\\multicolumn{3}{c}{Snake vs block} \\\\\n"
              "        \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}\n"
              "        Network & Block & Altern. & Snake & Smaller & Equal & Larger & Smaller & Equal & Larger \\\\")
    medians = ["med_sequential_a", "med_alternating_a", "med_snake"]
    counts = ["alternating_a_smaller", "alternating_a_equal", "alternating_a_larger", "snake_smaller", "snake_equal",
              "snake_larger"]
    body = [table_row([network] + [f"{balance.loc[network, c]:.2f}" for c in medians]
                      + [str(int(balance.loc[network, c])) for c in counts]) for network in NETWORKS]
    write_table("tab:results-selfplay-balance",
                float_table("tab:results-selfplay-balance", caption,
                            "@{}lrrr@{\\hspace{10pt}}rrr@{\\hspace{10pt}}rrr@{}", header, body,
                            short="Balance in deterministic self-play", after_label="\\setlength{\\tabcolsep}{3.5pt}"))


# Algorithms
def average() -> None:
    mean_margin = read_data("average.csv", index_col=0)
    positive = read_data("positive.csv", index_col=0)
    thresholds = len(POSITIVE_THETAS)
    means = 8 * thresholds
    caption = ("Average performance at \\(k=20\\) under \\seq{snake}. Left: mean own margin over the four opponents "
               f"in both roles, in percent of the network, averaged over the {thresholds} positive thresholds. Right: "
               f"number of the {means} opponent-and-role means (eight at each of the {thresholds} thresholds) that "
               "are positive. D: Degree; C: Coverage Greedy; P: PageRank; B: Betweenness; R: Random.")
    header = ("        & \\multicolumn{5}{c}{Mean own margin (\\%)} & \\multicolumn{5}{c}{Positive means (of "
              f"{means})}} \\\\\n"
              "        \\cmidrule(lr){2-6}\\cmidrule(l){7-11}\n"
              "        Network & D & C & P & B & R & D & C & P & B & R \\\\")
    body = [table_row([network] + [signed(mean_margin.loc[network, a], ".1f") for a in "DCPBR"]
                      + [str(positive.loc[network, a]) for a in "DCPBR"]) for network in NETWORKS]
    write_table("tab:results-average",
                float_table("tab:results-average", caption, "@{}lrrrrr@{\\hspace{10pt}}rrrrr@{}", header, body,
                            short="Average performance of the algorithms",
                            after_label="\\setlength{\\tabcolsep}{3.5pt}"))


def competitive() -> None:
    ordering = read_data("competitive_ordering.csv", index_col=0)
    caption = ("Algorithm with a positive mean margin against every other algorithm in both roles, at $k=20$ under "
               "\\seq{snake}. D: Degree; C: Coverage Greedy; P: PageRank; B: Betweenness; --: no algorithm satisfies "
               "the criterion. Each symbol requires all eight opponent-and-role means to be positive. The criterion "
               "concerns signs, not the size or statistical significance of a lead.")
    header = "        Network & " + " & ".join(theta_math(t) for t in POSITIVE_THETAS) + " \\\\"
    body = [table_row([network] + [ordering.loc[network, THETA_FRACTION[t]] for t in POSITIVE_THETAS])
            for network in NETWORKS]
    label = "tab:results-competitive-ordering"
    write_table(label, float_table(label, caption, "@{}l" + "c" * len(POSITIVE_THETAS) + "@{}", header, body,
                                   before_caption="\\setlength{\\tabcolsep}{3pt}"))


def elo_table(ratings: pd.DataFrame) -> None:
    overall = ratings[(ratings.cutoff == "std") & (ratings.regime == "ALL") & (ratings.seed == ELO_SEED)]
    elo = overall.pivot(index="network", columns="alg", values="elo")
    caption = ("Elo ratings of the five algorithms on each network, from all games between different algorithms at "
               f"the {len(POSITIVE_THETAS)} positive thresholds, all budgets, all sequences, and both roles, averaged "
               "over 200 random game orders. All algorithms start at 1500. The highest rating on each network is "
               "in bold.")
    header = "        Network & Degree & Coverage Greedy & PageRank & Betweenness & Random \\\\"
    body = []
    for network in NETWORKS:
        values = elo.loc[network, ALGORITHMS]
        best = values.idxmax()
        body.append(table_row([network] + [f"\\textbf{{{half_up(values[a])}}}" if a == best else half_up(values[a])
                                           for a in ALGORITHMS]))
    write_table("tab:results-elo", float_table("tab:results-elo", caption, "@{}lrrrrr@{}", header, body,
                                               short="Elo ratings of the algorithms"))


def winner(ratings: pd.DataFrame, regime: str, cutoff: str, network: str) -> dict:
    """Highest-rated algorithm (main seed), its lead over the second, and whether all seeds agree."""
    rows = ratings[(ratings.regime == regime) & (ratings.cutoff == cutoff) & (ratings.network == network)]
    main_seed = rows[rows.seed == ELO_SEED].set_index("alg")
    ranked = main_seed.elo.sort_values(ascending=False)
    winner_per_seed = rows.loc[rows.groupby("seed").elo.idxmax(), "alg"]
    return dict(win=ranked.index[0], lead=ranked.iloc[0] - ranked.iloc[1], stable=winner_per_seed.nunique() == 1,
                games=int(main_seed.games.iloc[0]), elo=main_seed.elo)


def regime_summary(ratings: pd.DataFrame, regime: str, cutoff: str) -> pd.DataFrame:
    """Mean rating per algorithm over the networks, and the number of networks on which it is first."""
    rows = ratings[(ratings.regime == regime) & (ratings.cutoff == cutoff) & (ratings.seed == ELO_SEED)]
    elo = rows.pivot(index="network", columns="alg", values="elo").loc[NETWORKS, ALGORITHMS]
    first = elo.idxmax(axis=1).value_counts().reindex(ALGORITHMS, fill_value=0)
    return pd.DataFrame({"mean": elo.mean(), "first": first})


def elo_regimes(ratings: pd.DataFrame, stats: pd.DataFrame) -> None:
    collapse = read_data("collapse.csv", index_col=0)
    caption = ("Elo ratings of the five algorithms on each network by threshold regime, assigned per network and "
               "budget: R1, \\(\\theta\\in\\{0,1/16\\}\\); R2, \\(\\theta>1/16\\) with a median adoption of at least "
               "half of the nodes over all cross-algorithm conditions of that network, budget and threshold; R3, the "
               "other \\(\\theta>1/16\\). All: every positive threshold (\\cref{tab:results-elo}). Each block rates "
               "all games between different algorithms in it (all budgets, sequences, and both roles), averaged over "
               "200 random game orders from 1500. Ratings are comparable only within a block, since the blocks hold "
               "different numbers of games. Bold: highest rating on the network in the block, and highest Mean. "
               "Mean: mean over the 11 networks. First: number of networks on which the algorithm is rated highest. "
               "Path: average shortest-path length (sampled). Last R2, First R3: thresholds at \\(k=20\\). "
               "D: Degree; C: Coverage Greedy; P: PageRank; B: Betweenness; R: Random. "
               "\\textsuperscript{\\ddag}Tie: the highest rating changes with the random game orders.")
    header = "        & & " + " & ".join(f"\\rotatebox{{90}}{{{n}}}" for n in NETWORKS) + " & Mean & First \\\\"
    body = ["        \\multicolumn{2}{@{}l}{Path} & "
            + " & ".join(f"{stats.loc[n, 'avg_path']:.2f}" for n in NETWORKS) + " & & \\\\",
            "        \\multicolumn{2}{@{}l}{Last R2} & "
            + " & ".join(theta_math(collapse.loc[n, "regime_last_R2"]) for n in NETWORKS) + " & & \\\\",
            "        \\multicolumn{2}{@{}l}{First R3} & "
            + " & ".join(theta_math(collapse.loc[n, "regime_first_R3"]) for n in NETWORKS) + " & & \\\\"]
    for regime in REGIMES:
        winners = {n: winner(ratings, regime, "std", n) for n in NETWORKS}
        summary = regime_summary(ratings, regime, "std")
        body += ["        \\midrule",
                 f"        {'All' if regime == 'ALL' else regime} & Games & "
                 + " & ".join(thousands(winners[n]["games"]) for n in NETWORKS) + " & & \\\\"]
        best_mean = summary["mean"].idxmax()
        for algorithm in ALGORITHMS:
            cells = ["", LETTER[algorithm]]
            for network in NETWORKS:
                w = winners[network]
                value = half_up(w["elo"][algorithm])
                if algorithm == w["win"]:
                    value = f"\\textbf{{{value}}}" + ("" if w["stable"] else "\\textsuperscript{\\ddag}")
                cells.append(value)
            mean = half_up(summary.loc[algorithm, "mean"])
            cells += [f"\\textbf{{{mean}}}" if algorithm == best_mean else mean,
                      str(int(summary.loc[algorithm, "first"]))]
            body.append("        " + " & ".join(cells) + " \\\\")
    label = "tab:results-elo-regimes"
    spec = "@{}ll" + "r" * len(NETWORKS) + "@{\\hspace{8pt}}rr@{}"
    write_table(label, float_table(label, caption, spec, header, body, short="Elo ratings by threshold regime",
                                   after_label="\\setlength{\\tabcolsep}{2.2pt}", size="footnotesize"))


def elo_robustness(ratings: pd.DataFrame) -> None:
    caption = ("Highest-rated algorithm by Elo on each network and threshold regime (\\cref{tab:results-elo-regimes}), "
               "with its lead over the second-rated algorithm in rating points. Std.: B's score is its win proportion "
               "plus half its draw proportion, as in \\cref{tab:results-elo}. 1\\%: a game counts as a win only if the "
               "mean margin exceeds 1\\% of the nodes in size, \\(|m_B|>0.01\\), and as a draw otherwise; for "
               "conditions with Random the mean over the 100 runs decides. Bold: the highest-rated algorithm differs "
               "from Std. Highest mean: algorithm with the highest mean rating over the 11 networks, with its lead "
               "over the second. Most often first: algorithm with the highest rating on the most networks, with the "
               "number of networks. D: Degree; C: Coverage Greedy; P: PageRank; B: Betweenness. "
               "\\textsuperscript{\\ddag}The highest rating changes with the random game orders (tie).")
    header = ("        & \\multicolumn{2}{c}{R1} & \\multicolumn{2}{c}{R2} & \\multicolumn{2}{c}{R3} & "
              "\\multicolumn{2}{c}{All} \\\\\n"
              "        \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}\n"
              "        Network & Std. & 1\\% & Std. & 1\\% & Std. & 1\\% & Std. & 1\\% \\\\")
    body = []
    for network in NETWORKS:
        cells = [network]
        for regime in REGIMES:
            standard = winner(ratings, regime, "std", network)
            one_percent = winner(ratings, regime, "1pc", network)
            for w in (standard, one_percent):
                cell = f"{LETTER[w['win']]} ({half_up(w['lead'])})"
                if not w["stable"]:
                    cell += "\\textsuperscript{\\ddag}"
                if w is one_percent and one_percent["win"] != standard["win"]:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
        body.append("        " + " & ".join(cells) + " \\\\")
    highest_mean, most_often_first = ["Highest mean"], ["Most often first"]
    for regime in REGIMES:
        for cutoff in ("std", "1pc"):
            summary = regime_summary(ratings, regime, cutoff)
            means = summary["mean"].sort_values(ascending=False)
            highest_mean.append(f"{LETTER[means.index[0]]} ({half_up(means.iloc[0] - means.iloc[1])})")
            first = summary["first"]
            leaders = [LETTER[a] for a in ALGORITHMS if first[a] == first.max()]
            most_often_first.append(f"{'/'.join(leaders)} ({int(first.max())})")
    body += ["        \\midrule", "        " + " & ".join(highest_mean) + " \\\\",
             "        " + " & ".join(most_often_first) + " \\\\"]
    label = "tab:results-elo-robustness"
    write_table(label, float_table(label, caption, "@{}lcccccccc@{}", header, body,
                                   short="Robustness of the Elo winners",
                                   after_label="\\setlength{\\tabcolsep}{2.5pt}"))


def main() -> None:
    stats = read_data("network_stats.csv").set_index("network")
    medians = read_data("medians.csv")
    medians["theta"] = medians.theta.round(6)
    snake = medians[medians.allocation_order == "snake"].set_index(["graph_name", "k", "theta"])
    examples = read_data("examples.csv")
    ratings = read_data("elo_all.csv")

    adoption(stats, snake)
    rounds(stats, medians)
    budget(stats, snake)
    appendix(snake)
    similarity()
    overlap()
    outcome_changes()
    priority_counterexample(examples)
    youtube_allocation(examples)
    threshold_reversal(examples)
    topk_share()
    selfplay()
    average()
    competitive()
    elo_table(ratings)
    elo_regimes(ratings, stats)
    elo_robustness(ratings)
    print(sorted(os.listdir(OUT_TABLES)))


if __name__ == "__main__":
    main()
