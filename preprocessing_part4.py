
import sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
SEED = 42
BLUE, ORANGE, VERM, GREY = "#0072B2", "#E69F00", "#D55E00", "#999999"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "figure.dpi": 150, "pdf.fonttype": 42})

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("part4_out")
fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
R = {}  # every number used in the report ends up in results.json


def save(fig, name):
    fig.savefig(fig_dir / name, bbox_inches="tight")
    plt.close(fig)


def names(text):
    """names inside a JSON-like column (genres, keywords, ...)"""
    try:
        return [item["name"] for item in json.loads(text)]
    except Exception:
        return []


# ---------------------------------------------------------------- 0. load and clean
movies = pd.read_csv(data_dir / "tmdb_5000_movies.csv")
credits = pd.read_csv(data_dir / "tmdb_5000_credits.csv")
df = movies.merge(credits[["movie_id", "cast"]], left_on="id", right_on="movie_id", how="left")

df["genre_list"] = df["genres"].apply(names)
df["n_genres"] = df["genre_list"].str.len()
df["n_keywords"] = df["keywords"].apply(lambda t: len(names(t)))
df["n_companies"] = df["production_companies"].apply(lambda t: len(names(t)))
df["n_countries"] = df["production_countries"].apply(lambda t: len(names(t)))
df["n_cast"] = df["cast"].apply(lambda t: len(json.loads(t)) if isinstance(t, str) else 0)

date = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = date.dt.year
df["release_month"] = date.dt.month

# cleaning rules of the group: zero runtime -> median, zero budget/revenue -> unknown
runtime = df["runtime"].where(df["runtime"] > 0)
df["runtime"] = runtime.fillna(runtime.median())
df["budget"] = df["budget"].where(df["budget"] > 0)
df["revenue"] = df["revenue"].where(df["revenue"] > 0)

# main data: films with a known budget and revenue
sub = df.dropna(subset=["budget", "revenue"]).copy()
sub["release_year"] = sub["release_year"].fillna(sub["release_year"].median())
sub["release_month"] = sub["release_month"].fillna(sub["release_month"].median())
R["n_all"], R["n_complete"] = len(df), len(sub)
R["runtime_median_used"] = float(runtime.median())
R["runtime_values_imputed"] = int(runtime.isna().sum())

# log scale for the skewed attributes
sub["log_budget"] = np.log10(sub["budget"])
sub["log_revenue"] = np.log10(sub["revenue"])
sub["log_popularity"] = np.log10(sub["popularity"])
sub["log_vote_count"] = np.log10(sub["vote_count"] + 1)

PRE = ["log_budget", "runtime", "release_year", "release_month", "n_genres",
       "n_keywords", "n_companies", "n_countries", "n_cast"]      # known before release
POST = ["log_popularity", "vote_average", "log_vote_count"]       # known only after release
LABEL = {"log_budget": "log10 budget", "runtime": "runtime", "release_year": "release year",
         "release_month": "release month", "n_genres": "# genres", "n_keywords": "# keywords",
         "n_companies": "# prod. companies", "n_countries": "# prod. countries",
         "n_cast": "# cast members", "log_popularity": "log10 popularity",
         "vote_average": "vote average", "log_vote_count": "log10 vote count",
         "log_revenue": "log10 revenue"}
y = sub["log_revenue"].values

# ---------------------------------------------------------------- 1. correlation matrix
cols = PRE + POST + ["log_revenue"]
corr = sub[cols].corr()

fig, ax = plt.subplots(figsize=(7.0, 5.3))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels([LABEL[c] for c in cols], rotation=45, ha="right")
ax.set_yticklabels([LABEL[c] for c in cols])
for i in range(len(cols)):
    for j in range(len(cols)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(v) > 0.6 else "black")
for k in (len(PRE) - 0.5, len(PRE) + len(POST) - 0.5):   # separate the three blocks
    ax.axhline(k, color="k", lw=0.8)
    ax.axvline(k, color="k", lw=0.8)
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("Pearson r")
save(fig, "fig4_corr.pdf")

R["corr_with_target"] = {c: float(corr.loc[c, "log_revenue"]) for c in PRE + POST}
pairs = sorted(((a, b, float(corr.loc[a, b])) for i, a in enumerate(PRE) for b in PRE[i + 1:]),
               key=lambda t: -abs(t[2]))
R["top_pre_release_pairs"] = pairs[:5]
R["pairs_above_0.8"] = [p for p in pairs if abs(p[2]) > 0.8]
R["corr_popularity_votecount"] = float(corr.loc["log_popularity", "log_vote_count"])
R["raw_pearson_budget_revenue"] = float(sub["budget"].corr(sub["revenue"]))

# ---------------------------------------------------------------- 2. attribute selection
# step 1: the post-release attributes are removed (data leakage)
# step 2: no pair of pre-release attributes has |r| > 0.8 (see above)
# step 3: mutual information with log revenue, compared with a noise level
X_all = sub[PRE + POST].values
mi = np.mean([mutual_info_regression(X_all, y, n_neighbors=3, random_state=s) for s in range(5)], axis=0)
MI = dict(zip(PRE + POST, mi.tolist()))

# noise level = biggest MI we get when the target is shuffled
rng = np.random.RandomState(SEED)
noise = max(mutual_info_regression(sub[PRE].values, rng.permutation(y), n_neighbors=3,
                                   random_state=s).max() for s in range(10))
strict = [c for c in PRE if MI[c] >= noise]
R["mi"], R["noise_level"], R["mi_above_noise"] = MI, float(noise), strict

fig, ax = plt.subplots(figsize=(3.4, 3.1))
order = sorted(MI, key=MI.get)
colors = [VERM if c in POST else BLUE if c in strict else GREY for c in order]
ax.barh([LABEL[c] for c in order], [MI[c] for c in order], color=colors)
ax.axvline(noise, color="k", ls="--", lw=0.8)
ax.set_xlabel("Mutual information with log10 revenue (nats)")
ax.legend(handles=[plt.Line2D([0], [0], color=BLUE, lw=6, label="above noise level"),
                   plt.Line2D([0], [0], color=GREY, lw=6, label="within noise level"),
                   plt.Line2D([0], [0], color=VERM, lw=6, label="post-release (removed)")],
          loc="lower right", frameon=False)
save(fig, "fig4_selection.pdf")

# does a strict filter hurt? 5-fold CV R2 for different feature sets
feature_sets = {"Only log10 budget": ["log_budget"],
                "Strict MI filter (4)": strict,
                "All pre-release features (9)": PRE,
                "Pre-release + post-release (12)": PRE + POST}
folds = KFold(5, shuffle=True, random_state=SEED)
R["selection_cv"] = {}
for name, feats in feature_sets.items():
    X = sub[feats].values
    ridge = cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), X, y, cv=folds, scoring="r2")
    forest = cross_val_score(RandomForestRegressor(200, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
                             X, y, cv=folds, scoring="r2")
    R["selection_cv"][name] = {"n_features": len(feats),
                               "ridge": [float(ridge.mean()), float(ridge.std())],
                               "rf": [float(forest.mean()), float(forest.std())]}
# the strict filter loses R2, so all 9 pre-release attributes are kept
FEATURES = PRE

# ---------------------------------------------------------------- 3. normalization / standardization
budget_musd = sub["budget"].values / 1e6
log_budget = sub["log_budget"].values
minmax = lambda v: MinMaxScaler().fit_transform(v.reshape(-1, 1)).ravel()
zscore = lambda v: StandardScaler().fit_transform(v.reshape(-1, 1)).ravel()

panels = [("(a) Min-max, raw budget", minmax(budget_musd), BLUE),
          ("(b) Min-max, log10 budget", minmax(log_budget), BLUE),
          ("(c) z-score, raw budget", zscore(budget_musd), ORANGE),
          ("(d) z-score, log10 budget", zscore(log_budget), ORANGE)]
fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.3))
for ax, (title, values, color) in zip(axs.ravel(), panels):
    ax.hist(values, bins=30, color=color, alpha=0.85)
    ax.set_title(title, fontsize=7)
    ax.set_ylabel("Films")
for ax in axs[1]:                       # z = -3 and z = +3
    ax.axvline(-3, color="k", ls=":", lw=0.8)
    ax.axvline(3, color="k", ls=":", lw=0.8)
fig.tight_layout()
save(fig, "fig4_scaling.pdf")

tiny = (sub["budget"] < 1e4) | (sub["revenue"] < 1e4)       # implausibly small values
R["scaling"] = {
    "skew_budget_raw": float(skew(sub["budget"])), "skew_budget_log": float(skew(log_budget)),
    "skew_revenue_raw": float(skew(sub["revenue"])), "skew_revenue_log": float(skew(y)),
    "skew_popularity_raw": float(skew(sub["popularity"])), "skew_popularity_log": float(skew(sub["log_popularity"])),
    "skew_votecount_raw": float(skew(sub["vote_count"])), "skew_votecount_log": float(skew(sub["log_vote_count"])),
    "minmax_raw_share_below_0.1": float((minmax(budget_musd) < 0.1).mean()),
    "minmax_log_share_below_0.1": float((minmax(log_budget) < 0.1).mean()),
    "minmax_raw_median": float(np.median(minmax(budget_musd))),
    "minmax_log_median": float(np.median(minmax(log_budget))),
    "z_raw_n_abs_gt3": int((np.abs(zscore(budget_musd)) > 3).sum()),
    "z_log_n_abs_gt3": int((np.abs(zscore(log_budget)) > 3).sum()),
    "z_raw_max": float(zscore(budget_musd).max()), "z_log_min": float(zscore(log_budget).min()),
    # same numbers again after removing the films with budget or revenue below 10,000 USD
    "n_tiny_removed_sensitivity": int(tiny.sum()),
    "skew_budget_log_no_tiny": float(skew(sub.loc[~tiny, "log_budget"])),
    "skew_revenue_log_no_tiny": float(skew(sub.loc[~tiny, "log_revenue"])),
    "z_log_min_no_tiny": float(zscore(sub.loc[~tiny, "log_budget"].values).min()),
}

# ---------------------------------------------------------------- 4. one-hot encoding
genres = sorted({g for names_ in sub["genre_list"] for g in names_})
genre_cols = pd.DataFrame({f"genre_{g}": sub["genre_list"].apply(lambda l: int(g in l)) for g in genres},
                          index=sub.index)                               # multi-hot
top_languages = sub["original_language"].value_counts().head(5).index.tolist()
language = sub["original_language"].where(sub["original_language"].isin(top_languages), "other")
language_cols = pd.get_dummies(language, prefix="lang").astype(int)      # one-hot
R["n_genre_cols"], R["n_language_cols"] = genre_cols.shape[1], language_cols.shape[1]
R["language_top5"] = sub["original_language"].value_counts().head(5).to_dict()
R["language_other_share"] = float((language == "other").mean())
R["genres_per_film_mean"] = float(sub["n_genres"].mean())
R["films_without_genre"] = int((sub["n_genres"] == 0).sum())

# ---------------------------------------------------------------- 5. PCA
X_num = StandardScaler().fit_transform(sub[FEATURES].values)             # z-scored numeric attributes
X_full = np.hstack([X_num, genre_cols.values, language_cols.values]).astype(float)
col_names = FEATURES + list(genre_cols.columns) + list(language_cols.columns)

pca_num = PCA().fit(X_num)
pca_full = PCA().fit(X_full)
cum_num = np.cumsum(pca_num.explained_variance_ratio_)
cum_full = np.cumsum(pca_full.explained_variance_ratio_)
Z = pca_full.transform(X_full)
loadings = pd.DataFrame(pca_full.components_[:3].T, index=col_names, columns=["PC1", "PC2", "PC3"])

R["pca"] = {
    "n_numeric": X_num.shape[1], "n_full": X_full.shape[1],
    "num_cum_2": float(cum_num[1]), "full_cum_2": float(cum_full[1]), "full_cum_3": float(cum_full[2]),
    "full_evr_first5": pca_full.explained_variance_ratio_[:5].tolist(),
    "k_80_num": int(np.searchsorted(cum_num, 0.80) + 1), "k_90_num": int(np.searchsorted(cum_num, 0.90) + 1),
    "k_80_full": int(np.searchsorted(cum_full, 0.80) + 1), "k_90_full": int(np.searchsorted(cum_full, 0.90) + 1),
    "loadings_top": {pc: loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)
                     .head(4).round(3).to_dict() for pc in loadings.columns},
    "corr_pc_logrev_full": [float(np.corrcoef(Z[:, i], y)[0, 1]) for i in range(3)],
}
R["pca"]["n_films_pc1_below_-5"] = int((Z[:, 0] < -5).sum())
R["pca"]["n_of_those_budget_below_10000"] = int((sub["budget"].values[Z[:, 0] < -5] < 1e4).sum())

fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(range(1, len(cum_num) + 1), cum_num, "o-", color=BLUE, ms=3, label=f"Numeric only ({X_num.shape[1]} features)")
ax.plot(range(1, len(cum_full) + 1), cum_full, "s-", color=ORANGE, ms=2.5,
        label=f"Numeric + one-hot ({X_full.shape[1]} features)")
ax.axhline(0.9, color="k", ls=":", lw=0.8)
ax.set_xlabel("Number of principal components")
ax.set_ylabel("Cumulative explained variance")
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", frameon=False)
save(fig, "fig4_pca_scree.pdf")

fig = plt.figure(figsize=(7.0, 3.1))
grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.25)
ax2d = fig.add_subplot(grid[0, 0])
sc = ax2d.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=5, alpha=0.7, rasterized=True)
ax2d.set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0] * 100:.1f}%)")
ax2d.set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1] * 100:.1f}%)")
ax2d.set_title("(a) 2D projection")
ax3d = fig.add_subplot(grid[0, 1], projection="3d")
ax3d.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=y, cmap="viridis", s=4, alpha=0.6, rasterized=True)
ax3d.set_xlabel("PC1", labelpad=-4)
ax3d.set_ylabel("PC2", labelpad=-4)
ax3d.set_zlabel("PC3", labelpad=-4)
ax3d.tick_params(labelsize=6, pad=-2)
ax3d.set_title("(b) 3D projection")
fig.colorbar(sc, cax=fig.add_subplot(grid[0, 2])).set_label("log10 revenue (USD)")
save(fig, "fig4_pca_scatter.pdf")

# ---------------------------------------------------------------- 6. data augmentation (SMOTE)
# imbalanced label: blockbuster = revenue of at least 300 million USD
blockbuster = (sub["revenue"] >= 300e6).astype(int).values
X_aug = sub[FEATURES].values
R["aug"] = {"threshold": 300e6, "n_pos": int(blockbuster.sum()),
            "n_neg": int((1 - blockbuster).sum()), "share_pos": float(blockbuster.mean())}

cv_aug = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
metrics = {"f1": "f1", "recall": "recall", "precision": "precision", "roc_auc": "roc_auc"}
classifiers = {
    "LR": lambda w=None: LogisticRegression(max_iter=2000, class_weight=w),
    "RF": lambda w=None: RandomForestClassifier(200, min_samples_leaf=3, class_weight=w,
                                                random_state=SEED, n_jobs=-1)}
R["aug"]["cv"] = {}
for kind, make in classifiers.items():
    strategies = {  # SMOTE sits inside the pipeline, so it only touches the training folds
        "Baseline": ImbPipeline([("scale", StandardScaler()), ("clf", make())]),
        "Class weights": ImbPipeline([("scale", StandardScaler()), ("clf", make("balanced"))]),
        "SMOTE": ImbPipeline([("scale", StandardScaler()), ("smote", SMOTE(k_neighbors=5, random_state=SEED)),
                              ("clf", make())])}
    for strategy, pipe in strategies.items():
        res = cross_validate(pipe, X_aug, blockbuster, cv=cv_aug, scoring=metrics)
        R["aug"]["cv"][f"{kind}|{strategy}"] = {m: [float(res[f"test_{m}"].mean()), float(res[f"test_{m}"].std())]
                                                for m in metrics}

# picture only: SMOTE on all films (the cross-validation above is the real test)
X_scaled = StandardScaler().fit_transform(X_aug)
X_res, y_res = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(X_scaled, blockbuster)
R["aug"]["after_smote_pos"], R["aug"]["after_smote_neg"] = int(y_res.sum()), int((1 - y_res).sum())
fx, fy = sorted(FEATURES, key=lambda c: -MI[c])[:2]                      # two most informative attributes
ix, iy = FEATURES.index(fx), FEATURES.index(fy)
R["aug"]["scatter_features"] = [fx, fy]

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.5]})
ax = axs[0]
ax.bar(["Original", "After SMOTE"], [blockbuster.sum(), y_res.sum()], color=ORANGE, label="Blockbuster (>= $300M)")
ax.bar(["Original", "After SMOTE"], [(1 - blockbuster).sum(), (1 - y_res).sum()],
       bottom=[blockbuster.sum(), y_res.sum()], color=BLUE, label="Other")
ax.set_ylabel("Films")
ax.set_ylim(0, 7800)
ax.set_title("(a) Class counts")
ax.legend(frameon=False, loc="upper left")
ax = axs[1]
ax.scatter(X_scaled[blockbuster == 0, ix], X_scaled[blockbuster == 0, iy], s=4, c=GREY, alpha=0.35,
           label="Other", rasterized=True)
ax.scatter(X_scaled[blockbuster == 1, ix], X_scaled[blockbuster == 1, iy], s=9, c=ORANGE,
           label="Real blockbuster", rasterized=True)
ax.scatter(X_res[len(X_scaled):, ix], X_res[len(X_scaled):, iy], s=9, c=VERM, marker="x", lw=0.6,
           label="Synthetic (SMOTE)", rasterized=True)
ax.set_xlabel(f"{LABEL[fx]} (z-score)")
ax.set_ylabel(f"{LABEL[fy]} (z-score)")
ax.set_title("(b) Synthetic samples in feature space")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig4_smote.pdf")

# ---------------------------------------------------------------- 7. imputation of budget
# Revenue is the target, so it is never imputed and never used as an input.
full = df.copy()
full["release_year"] = full["release_year"].fillna(full["release_year"].median())
full["release_month"] = full["release_month"].fillna(full["release_month"].median())
full["log_budget"] = np.log10(full["budget"])
full.loc[full["budget"] < 1e4, "log_budget"] = np.nan                    # implausible budgets are no donors

genre_all = pd.DataFrame({g: full["genre_list"].apply(lambda l: int(g in l))
                          for g in sorted({g for names_ in full["genre_list"] for g in names_})}, index=full.index)
inputs = pd.concat([full[["runtime", "release_year", "release_month", "n_genres",
                          "n_companies", "n_countries", "n_cast"]], genre_all], axis=1)
inputs = pd.DataFrame(StandardScaler().fit_transform(inputs), index=full.index, columns=inputs.columns)
known = full.index[full["log_budget"].notna()]
decade = full["release_year"] // 10 * 10
R["imp"] = {"n_budget_unknown": int(full["budget"].isna().sum()),
            "n_budget_unknown_rev_known": int((full["budget"].isna() & full["revenue"].notna()).sum()),
            "n_donors": int(len(known))}


def knn_impute(frame_index, hidden=()):
    """KNN imputation of log budget for the rows in `hidden` (and any other missing row)."""
    data = inputs.loc[frame_index].copy()
    data["log_budget"] = full.loc[frame_index, "log_budget"].values
    data.loc[list(hidden), "log_budget"] = np.nan
    filled = KNNImputer(n_neighbors=5).fit_transform(data.values)
    return pd.Series(filled[:, -1], index=frame_index)


# masking test: hide 20% of the known budgets and see how well each method guesses them
errors = {m: [] for m in ["Mean", "Median", "Decade median", "KNN (k=5)"]}
for rep in range(10):
    hide = np.random.RandomState(100 + rep).rand(len(known)) < 0.2
    test, train = known[hide], known[~hide]
    truth = full.loc[test, "log_budget"].values
    decade_median = full.loc[train].groupby(decade[train])["log_budget"].median()
    guesses = {"Mean": np.full(len(test), full.loc[train, "log_budget"].mean()),
               "Median": np.full(len(test), full.loc[train, "log_budget"].median()),
               "Decade median": decade[test].map(decade_median).fillna(full.loc[train, "log_budget"].median()).values,
               "KNN (k=5)": knn_impute(known, hidden=test).loc[test].values}
    for m, guess in guesses.items():
        errors[m].append((np.sqrt(np.mean((guess - truth) ** 2)), np.mean(np.abs(guess - truth))))
R["imp"]["eval"] = {m: {"rmse": [float(np.mean([e[0] for e in v])), float(np.std([e[0] for e in v]))],
                        "mae": [float(np.mean([e[1] for e in v])), float(np.std([e[1] for e in v]))]}
                    for m, v in errors.items()}
best = min(R["imp"]["eval"], key=lambda m: R["imp"]["eval"][m]["rmse"][0])
R["imp"]["best"] = best

# real imputation: films with unknown budget but known revenue
to_fill = full.index[full["budget"].isna() & full["revenue"].notna()]
filled_log = knn_impute(full.index).loc[to_fill]
observed = 10 ** full["log_budget"].dropna() / 1e6
imputed = 10 ** filled_log / 1e6
R["imp"].update({"n_imputed": int(len(to_fill)), "n_usable_after": int(len(sub) + len(to_fill)),
                 "imputed_median_budget_musd": float(imputed.median()),
                 "observed_median_budget_musd": float(observed.median()),
                 "imputed_budget_q": [float(q) for q in np.percentile(imputed, [25, 50, 75])],
                 "observed_budget_q": [float(q) for q in np.percentile(observed, [25, 50, 75])]})

fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.3), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axs[0]
methods = list(R["imp"]["eval"])
ax.bar(range(4), [R["imp"]["eval"][m]["rmse"][0] for m in methods],
       yerr=[R["imp"]["eval"][m]["rmse"][1] for m in methods],
       color=[ORANGE if m == best else BLUE for m in methods], capsize=2)
ax.set_xticks(range(4))
ax.set_xticklabels(["Mean", "Median", "Decade\nmedian", "KNN"], fontsize=5.5, rotation=30, ha="right")
ax.set_ylabel("RMSE of log10 budget")
ax.set_title("(a) Masking test", fontsize=7)
ax = axs[1]
ax.hist(full["log_budget"].dropna(), bins=30, density=True, alpha=0.6, color=BLUE, label="Observed")
ax.hist(filled_log.values, bins=20, density=True, alpha=0.7, color=ORANGE, label="Imputed")
ax.set_xlabel("log10 budget")
ax.set_title("(b) Distribution", fontsize=7)
ax.legend(frameon=False, fontsize=6)
fig.tight_layout()
save(fig, "fig4_imputation.pdf")

# ---------------------------------------------------------------- save all numbers
(out_dir / "results.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
print("done:", len(list(fig_dir.glob("*.pdf"))), "figures, results.json written to", out_dir)
