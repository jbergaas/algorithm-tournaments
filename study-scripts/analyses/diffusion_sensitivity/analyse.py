"""What the variants change: ties, collapse, regime winners, first-pick advantage, predictors.

Every quantity is computed for each variant and draw separately and then summarised over the draws.
  adoption              share of the nodes owned at the end, seeds included
  median                median adoption over the 48 matches of different algorithms of a unit
  collapse              first sampled threshold where the median is below 1/2
  regimes               low (theta <= 1/16), spreading (median >= 1/2) or post-collapse
  decisive score        mean score in a regime, with margins of at most 1% of the nodes as draws
  first-pick advantage  first picker's margin averaged over the two role assignments of a pair
  before the collapse   median >= 1/2 at theta > 0
  distance race         share of decided matches at theta <= 1/16 won by the player with more
                        nodes strictly nearer to its seeds
  first-round capture   share of decided spreading-regime comparisons won by the algorithm whose
                        own top k, playing alone, wins more nodes in the first round

Reads out/units/*.npz (simulate.py) and out/race_match.csv (race.py). Writes out/summary.txt (also
printed), out/ties_by_network.csv, out/collapse.csv, out/winners.csv, out/first_pick.csv and
out/predictors.csv.
Usage: python3 diffusion_sensitivity/analyse.py [networks...]   (default: all 15)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import variants as v

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s  # noqa: E402

OUT = s.out_dir("diffusion_sensitivity")
CUT = 0.01
VARIANT = {name: i for i, name in enumerate(v.VARIANT_NAMES)}
THRESHOLD_VARIANTS = ["hom-1%", "hom+1%", "U0.01", "U0.10", "U0.25", "U0.50", "D3"]
SHORT = [s.SHORT[a] for a in s.STRUCTURAL]
MATCH = pd.DataFrame(v.MATCHES, columns=["a", "b", "seq"]).assign(mi=np.arange(len(v.MATCHES)))
MATCH["cross"] = MATCH.a != MATCH.b
LINES: list[str] = []


def say(text: str = "") -> None:
    LINES.append(text)
    print(text)


def load_units(networks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All replayed matches and all first-round captures, one row each."""
    matches, capture = [], []
    for network in networks:
        for k in v.BUDGETS:
            for ti in range(len(v.THETAS)):
                z = np.load(OUT / "units" / f"{network}_k{k}_t{ti}.npz")
                key = dict(network=network, k=k, ti=ti)
                m = pd.DataFrame(z["matches"], columns=["vi", "draw", "mi", "spread_a", "spread_b", "tied"])
                matches.append(m.assign(**key, n=int(z["n"])))
                capture.append(pd.DataFrame(z["capture"], columns=["ai", "vi", "draw", "captured"]).assign(**key))
    m = pd.concat(matches, ignore_index=True).merge(MATCH, on="mi", how="left")
    m["margin_b"] = (m.spread_b - m.spread_a) / m.n
    m["adoption"] = (m.spread_a + m.spread_b) / m.n
    return m, pd.concat(capture, ignore_index=True)


def game_score(margin: np.ndarray, cut: float) -> np.ndarray:
    return np.select([margin > cut, margin < -cut], [1.0, 0.0], 0.5)


def regimes(m: pd.DataFrame) -> pd.DataFrame:
    """Median adoption and regime of every variant, draw, network, budget and threshold."""
    med = m[m.cross].groupby(["vi", "draw", "network", "k", "ti"]).adoption.median().rename("median").reset_index()
    med["regime"] = np.where(med.ti <= 1, "low", np.where(med["median"] >= 0.5, "spread", "post"))
    return med


def ties(m: pd.DataFrame) -> None:
    b = m[m.vi == VARIANT["base"]].assign(share=lambda x: x.tied / x.n)
    say("Ties in the unchanged model (all 64 matches of every unit)")
    say(f"  matches with at least one node neutral only because of a tie: {100 * (b.tied > 0).mean():.1f}%")
    say(f"  share of the network such nodes make up: mean {100 * b.share.mean():.2f}%, "
        f"max {100 * b.share.max():.2f}% ({b.loc[b.share.idxmax(), 'network']})")
    per = b.groupby("network").agg(any_tied=("tied", lambda x: (x > 0).mean()),
                                   mean_pct=("share", lambda x: 100 * x.mean()),
                                   max_pct=("share", lambda x: 100 * x.max()), max_nodes=("tied", "max"))
    per.reindex(v.NETWORKS).to_csv(OUT / "ties_by_network.csv")


def collapse(med: pd.DataFrame) -> None:
    rows = []
    for (vi, d, network, k), x in med.groupby(["vi", "draw", "network", "k"]):
        values = x.sort_values("ti")["median"].to_numpy()
        below = np.flatnonzero(values < 0.5)
        rows.append(dict(vi=vi, draw=d, network=network, k=k, collapse=int(below[0]) if below.size else len(values)))
    col = pd.DataFrame(rows)
    col.assign(variant=col.vi.map(dict(enumerate(v.VARIANT_NAMES)))).to_csv(OUT / "collapse.csv", index=False)
    base = col[col.vi == VARIANT["base"]][["network", "k", "collapse"]].rename(columns={"collapse": "base"})
    col = col.merge(base, on=["network", "k"])
    col["shift"] = col.collapse - col.base
    say("\nCollapse interval against the unchanged model, over networks x budgets x draws (sampled intervals)")
    say("  variant  mean shift   same  one earlier  one later  more   (one earlier: cases)")
    for vi, x in col.groupby("vi"):
        shift = x["shift"]
        say(f"  {v.VARIANT_NAMES[vi]:7s} {shift.mean():+10.3f}  {(shift == 0).mean():5.3f}"
            f"  {(shift == -1).mean():11.3f}  {(shift == 1).mean():9.3f}  {(shift.abs() > 1).mean():5.3f}"
            f"   ({int((shift == -1).sum())} of {len(x)})")


def decisive_scores(games: pd.DataFrame, med: pd.DataFrame) -> pd.DataFrame:
    """Decisive score of each algorithm per variant, draw, network and regime."""
    g = games.merge(med[["vi", "draw", "network", "k", "ti", "regime"]], on=["vi", "draw", "network", "k", "ti"])
    rows = []
    for (vi, d, network, regime), x in g.groupby(["vi", "draw", "network", "regime"], sort=False):
        a = x.a.map(s.STRUCTURAL.index).to_numpy()
        b = x.b.map(s.STRUCTURAL.index).to_numpy()
        sb = game_score(x.margin_b.to_numpy(), CUT)
        total = np.bincount(b, weights=sb, minlength=4) + np.bincount(a, weights=1 - sb, minlength=4)
        count = np.bincount(b, minlength=4) + np.bincount(a, minlength=4)
        rows.append(dict(vi=vi, draw=d, network=network, regime=regime, **dict(zip(SHORT, total / count))))
    return pd.DataFrame(rows)


def first_place(values: np.ndarray) -> frozenset:
    return frozenset(np.flatnonzero(values >= values.max() - 1e-9).tolist())


def winners(scores: pd.DataFrame) -> pd.DataFrame:
    """Per network, regime and variant: the share of draws whose first place is the unchanged
    model's winner (credit split on ties) and whether the first place of the draw-mean score lies
    within the unchanged model's winners."""
    out = []
    for (network, regime), x in scores.groupby(["network", "regime"]):
        if not (x.vi == VARIANT["base"]).any():
            continue
        base = first_place(x[x.vi == VARIANT["base"]][SHORT].to_numpy()[0])
        for name in v.VARIANT_NAMES[1:]:
            y = x[x.vi == VARIANT[name]]
            if y.empty:
                continue
            credit = [len(first_place(row) & base) / len(first_place(row)) for row in y[SHORT].to_numpy()]
            mean = first_place(y[SHORT].mean().to_numpy())
            out.append(dict(network=network, regime=regime, variant=name, draws=len(y),
                            base_winner="/".join(SHORT[i] for i in sorted(base)), kept=float(np.mean(credit)),
                            mean_winner="/".join(SHORT[i] for i in sorted(mean)), mean_kept=mean <= base))
    return pd.DataFrame(out)


def report_winners(w: pd.DataFrame) -> None:
    w.to_csv(OUT / "winners.csv", index=False)
    say("\nRegime winners by decisive score (share of draws keeping the unchanged model's winner)")
    for regime, label in (("low", "low-threshold"), ("spread", "spreading"), ("post", "post-collapse")):
        x = w[w.regime == regime]
        real = x.network.isin(v.REAL)
        say(f"  {label} regime: lowest share on the real networks {x[real].kept.min():.3f}, "
            f"on the generated networks {x[~real].kept.min():.3f}")
        for r in x[x.kept < 0.9].itertuples():
            say(f"    {r.network:14s} {r.variant:7s} {r.kept:.3f} (winner {r.base_winner}, draw-mean {r.mean_winner})")
        changed = x[~x.mean_kept]
        say(f"    draw-mean winner changed on the real networks: {int((~x[real].mean_kept).sum())}; on the generated: "
            + (", ".join(f"{r.network} {r.variant}" for r in changed[~changed.network.isin(v.REAL)].itertuples())
               or "none"))


def spreading_thresholds(med: pd.DataFrame, network: str) -> str:
    x = med[(med.vi == VARIANT["base"]) & (med.network == network) & (med.regime == "spread")]
    return "; ".join(f"k={k}: " + ", ".join(s.THETA_LABEL[v.THETAS[ti]] for ti in sorted(y.ti))
                     for k, y in x.groupby("k"))


def first_pick(games: pd.DataFrame, med: pd.DataFrame) -> None:
    keys = ["vi", "draw", "network", "k", "ti", "seq"]
    x = games[games.ti > 0][keys + ["a", "b", "margin_b"]]
    x = x[x.a < x.b].merge(x[x.a > x.b].rename(columns={"a": "b", "b": "a", "margin_b": "margin_reverse"}),
                           on=keys + ["a", "b"])
    sign = np.where(x.seq == "sequential_b", -1.0, 1.0)
    x["fp"] = (-sign * x.margin_b.to_numpy() - sign * x.margin_reverse.to_numpy()) / 2
    x = x.merge(med[["vi", "draw", "network", "k", "ti", "median"]], on=["vi", "draw", "network", "k", "ti"])
    x["before"] = x["median"] >= 0.5
    per_draw = x.groupby(["vi", "draw", "seq", "before"]).fp.median().mul(100).rename("median_fp").reset_index()
    t = per_draw.groupby(["vi", "seq", "before"]).median_fp.agg(["mean", "min", "max"]).reset_index()
    t["variant"] = t.vi.map(dict(enumerate(v.VARIANT_NAMES)))
    t.to_csv(OUT / "first_pick.csv", index=False)
    say("\nMedian first-pick advantage under the A-first block (% of nodes, mean over draws)")
    a = t[(t.seq == "sequential_a")].set_index(["variant", "before"])["mean"]
    for name in v.VARIANT_NAMES:
        say(f"  {name:7s} before the collapse {a[(name, True)]:6.2f}   after {a[(name, False)]:5.2f}")
    before = a.loc[THRESHOLD_VARIANTS].xs(True, level="before")
    after = a.loc[THRESHOLD_VARIANTS].xs(False, level="before")
    say(f"  threshold variants: {before.min():.1f} to {before.max():.1f} before, "
        f"{after.min():.2f} to {after.max():.2f} after")


def race(m: pd.DataFrame) -> pd.DataFrame:
    r = pd.read_csv(OUT / "race_match.csv")
    g = m[m.cross & (m.ti <= 1)][["vi", "draw", "network", "k", "ti", "mi", "margin_b"]]
    g = g.merge(r, on=["network", "k", "mi"])
    g = g[(g.margin_b.abs() > CUT) & (g.race_b != 0)]
    g["hit"] = np.sign(g.margin_b) == np.sign(g.race_b)
    t = g.groupby(["vi", "draw", "ti"]).hit.mean().groupby(["vi", "ti"]).mean().unstack("ti")
    say("\nDistance race: share of decided matches won by the race leader (mean over draws)")
    say("  variant  theta=0  theta=1/16")
    for vi, row in t.iterrows():
        say(f"  {v.VARIANT_NAMES[vi]:7s} {row[0]:7.3f}  {row[1]:9.3f}")
    say(f"  range over variants and both thresholds: {t.min().min():.3f} to {t.max().max():.3f}")
    return pd.DataFrame({"race_theta_0": t[0], "race_theta_1/16": t[1]})


def capture(games: pd.DataFrame, med: pd.DataFrame, cap: pd.DataFrame) -> pd.Series:
    keys = ["vi", "draw", "network", "k", "ti"]
    h = games.copy()
    sb = game_score(h.margin_b.to_numpy(), CUT)
    h["x"] = np.where(h.a < h.b, h.a, h.b)
    h["y"] = np.where(h.a < h.b, h.b, h.a)
    h["sx"] = np.where(h.a == h.x, 1 - sb, sb)
    cell = h.groupby(keys + ["x", "y"]).sx.mean().rename("h2h").reset_index()
    cell = cell.merge(med[keys + ["regime"]], on=keys)
    cell = cell[cell.regime == "spread"]
    cap = cap.assign(alg=cap.ai.map(dict(enumerate(s.STRUCTURAL))))[keys + ["alg", "captured"]]
    base = cap[cap.vi == VARIANT["base"]].drop(columns=["vi", "draw"])
    tie = pd.concat([base.assign(vi=VARIANT["tie"], draw=d) for d in range(v.DRAWS)])
    cap = pd.concat([cap, tie[cap.columns]])
    for side in ("x", "y"):
        cell = cell.merge(cap.rename(columns={"alg": side, "captured": f"c_{side}"}), on=keys + [side], how="left")
    assert cell.c_x.notna().all() and cell.c_y.notna().all()
    cell["d"] = cell.c_x - cell.c_y
    z = cell[(cell.h2h != 0.5) & (cell.d != 0)].copy()
    z["hit"] = np.sign(z.h2h - 0.5) == np.sign(z.d)
    t = z.groupby(["vi", "draw"]).hit.mean().groupby("vi").mean()
    say("\nFirst-round capture of the own top k: share of decided comparisons in the spreading regime won by "
        "the algorithm that captures more (mean over draws)")
    for vi, value in t.items():
        say(f"  {v.VARIANT_NAMES[vi]:7s} {value:.3f}")
    return t.rename("capture")


def main() -> None:
    m, cap = load_units(sys.argv[1:] or v.NETWORKS)
    med = regimes(m)
    games = m[m.cross][["vi", "draw", "network", "k", "ti", "a", "b", "seq", "margin_b"]]
    ties(m)
    collapse(med)
    report_winners(winners(decisive_scores(games, med)))
    say(f"  spreading thresholds of soc-Epinions1 in the unchanged model: {spreading_thresholds(med, 'soc-Epinions1')}")
    first_pick(games, med)
    predictors = race(m).join(capture(games, med, cap))
    predictors.index = [v.VARIANT_NAMES[i] for i in predictors.index]
    predictors.rename_axis("variant").to_csv(OUT / "predictors.csv")
    (OUT / "summary.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
