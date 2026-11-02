
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

NUM_COLS = ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]
LABELS = {
    "budget": "Budget (M USD)",
    "revenue": "Revenue (M USD)",
    "runtime": "Runtime (min)",
    "popularity": "Popularity",
    "vote_average": "Vote average",
    "vote_count": "Vote count",
}
MILLIONS = ["budget", "revenue"]
LOG_COLS = ["budget", "revenue", "popularity", "vote_count"]  # right-skewed


def load(path):
    df = pd.read_csv(path)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    return df


def scaled(df, col):
    """Return the column, in millions for money attributes."""
    return df[col] / 1e6 if col in MILLIONS else df[col]


# --------------------------------------------------------------------------
# Summary / IQR outliers
# --------------------------------------------------------------------------
def iqr_summary(df, name):
    rows = []
    for c in NUM_COLS:
        x = scaled(df, c).dropna()
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((x < lo) | (x > hi)).sum())
        rows.append({
            "subset": name, "attribute": c, "n": len(x),
            "Q1": q1, "median": med, "Q3": q3, "IQR": iqr,
            "upper_fence": hi, "n_outliers": n_out,
            "pct_outliers": 100 * n_out / len(x),
            "skewness": stats.skew(x),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Box plots
# --------------------------------------------------------------------------
def boxplots(df, title, fname, log=False):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, c in zip(axes.ravel(), NUM_COLS):
        x = scaled(df, c).dropna()
        use_log = log and c in LOG_COLS
        if use_log:
            x = np.log10(x[x > 0])
        ax.boxplot(x, vert=True, showfliers=True, widths=0.5,
                   flierprops=dict(marker="o", markersize=2, alpha=0.4),
                   patch_artist=True, boxprops=dict(facecolor="#9ecae1"),
                   medianprops=dict(color="darkred"))
        ax.set_title(("log10 " if use_log else "") + LABELS[c], fontsize=10)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Histograms
# --------------------------------------------------------------------------
def histograms(df, fname):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    panels = [
        ("runtime", df.loc[df.runtime > 0, "runtime"], "Runtime (min), runtime > 0", None),
        ("vote_average", df.loc[df.vote_count >= 10, "vote_average"],
         "Vote average, vote_count >= 10", None),
        ("release_year", df["release_year"].dropna(), "Release year", None),
        ("budget", np.log10(df.loc[df.budget > 0, "budget"]), "log10 Budget (USD), budget > 0", None),
        ("revenue", np.log10(df.loc[df.revenue > 0, "revenue"]), "log10 Revenue (USD), revenue > 0", None),
        ("vote_count", np.log10(df.loc[df.vote_count > 0, "vote_count"]), "log10 Vote count, > 0", None),
    ]
    for ax, (c, x, xl, _) in zip(axes.ravel(), panels):
        ax.hist(x, bins=30, color="#6baed6", edgecolor="white")
        ax.axvline(x.median(), color="darkred", ls="--", lw=1, label=f"median={x.median():.2f}")
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel("Films")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Scatter plots
# --------------------------------------------------------------------------
def scatter_pair(df, xcol, ycol, fname, suptitle, annotate_top=3):
    d = df[(df[xcol] > 0) & (df[ycol] > 0)].copy()
    x, y = d[xcol], d[ycol]

    # statistics
    pear_r, pear_p = stats.pearsonr(x, y)
    spear_r, spear_p = stats.spearmanr(x, y)
    lx, ly = np.log10(x), np.log10(y)
    ll_r, _ = stats.pearsonr(lx, ly)
    fit = stats.linregress(lx, ly)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # (a) linear scale
    xs = x / 1e6 if xcol in MILLIONS else x
    ys = y / 1e6 if ycol in MILLIONS else y
    a1.scatter(xs, ys, s=8, alpha=0.4, color="#3182bd")
    a1.set_xlabel(LABELS[xcol]); a1.set_ylabel(LABELS[ycol])
    a1.set_title("(a) Linear scale", fontsize=10)
    a1.grid(alpha=0.3)
    for _, r in d.nlargest(annotate_top, ycol).iterrows():
        xv = r[xcol] / 1e6 if xcol in MILLIONS else r[xcol]
        yv = r[ycol] / 1e6 if ycol in MILLIONS else r[ycol]
        a1.annotate(r["title"], (xv, yv), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")

    # (b) log-log with OLS line
    a2.scatter(x, y, s=8, alpha=0.4, color="#3182bd")
    gx = np.linspace(lx.min(), lx.max(), 100)
    a2.plot(10 ** gx, 10 ** (fit.intercept + fit.slope * gx), color="darkred",
            lw=1.5, label=f"log-log fit: slope={fit.slope:.2f}, r={ll_r:.2f}")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel(LABELS[xcol].replace("(M USD)", "(USD)") + " [log]")
    a2.set_ylabel(LABELS[ycol].replace("(M USD)", "(USD)") + " [log]")
    a2.set_title("(b) Log-log scale", fontsize=10)
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3, which="both")

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)

    out = {
        "pair": f"{xcol} vs {ycol}", "n": len(d),
        "pearson_r": pear_r, "spearman_rho": spear_r,
        "loglog_pearson_r": ll_r, "loglog_slope": fit.slope,
        "loglog_R2": fit.rvalue ** 2,
    }
    if xcol == "budget" and ycol == "revenue":
        out["pct_revenue_below_budget"] = 100 * (y < x).mean()
        out["pct_revenue_over_10x_budget"] = 100 * (y > 10 * x).mean()
        out["median_revenue_to_budget"] = (y / x).median()
    return out, d


def main():
    # Resolve paths relative to where this script file lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(base_dir, "data", "tmdb_5000_movies.csv")
    default_figures = os.path.join(base_dir, "figures")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=default_csv)
    ap.add_argument("--out", default=default_figures)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load(args.data)
    pos = df[(df.budget > 0) & (df.revenue > 0)]
    print(f"All records: {len(df)}   positive budget & revenue: {len(pos)}")

    # ---- summaries
    summ = pd.concat([iqr_summary(df, "all"), iqr_summary(pos, "positive")])
    summ.to_csv(os.path.join(args.out, "numeric_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n=== Numeric summary (money in millions) ===")
    print(summ.round(2).to_string(index=False))

    # ---- box plots
    boxplots(df, "Box plots of numeric attributes (all 4,803 films)",
             os.path.join(args.out, "fig_boxplots_all.png"))
    boxplots(pos, "Box plots, films with positive budget and revenue (log scale for skewed attributes)",
             os.path.join(args.out, "fig_boxplots_positive.png"), log=True)

    # ---- histograms
    histograms(df, os.path.join(args.out, "fig_histograms.png"))

    # ---- scatter plots
    rows = []
    r1, d1 = scatter_pair(df, "budget", "revenue",
                          os.path.join(args.out, "fig_scatter_budget_revenue.png"),
                          "Budget vs revenue (positive values only)")
    rows.append(r1)
    r2, d2 = scatter_pair(df, "vote_count", "revenue",
                          os.path.join(args.out, "fig_scatter_votecount_revenue.png"),
                          "Vote count vs revenue (positive values only; vote_count is post-release)")
    rows.append(r2)
    sc = pd.DataFrame(rows)
    sc.to_csv(os.path.join(args.out, "scatter_stats.csv"), index=False)
    print("\n=== Scatter statistics ===")
    print(sc.T.to_string(header=False))

    # extra facts that are handy for the discussion
    print("\nTop 5 revenue films:")
    print(df.nlargest(5, "revenue")[["title", "budget", "revenue", "vote_count"]].to_string(index=False))
    print("\nTop 5 by revenue/budget ratio (positive budget >= 10,000):")
    t = pos[pos.budget >= 10_000].assign(ratio=lambda z: z.revenue / z.budget)
    print(t.nlargest(5, "ratio")[["title", "budget", "revenue", "ratio"]].to_string(index=False))
    print("\nBottom 5 by revenue/budget ratio (positive budget >= 10,000):")
    print(t.nsmallest(5, "ratio")[["title", "budget", "revenue", "ratio"]].to_string(index=False))
    print("\nDone. Figures saved to", args.out)


if __name__ == "__main__":
    main()