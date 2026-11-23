"""CE 477 Assignment #2 - Regression (steps 1-3: targets, 70/30 split, models).
Both targets are NEW numeric attributes derived from the data set (not existing columns):
  1) log10 return on investment = log10(revenue / budget)
  2) weighted rating = Bayesian-shrunk vote_average (IMDb formula)

Input : tmdb_5000_movies.csv, tmdb_5000_credits.csv  (Kaggle TMDB 5000)
Output: regression_out/regression_results.json, regression_out/regression_predictions.csv
"""
import json
import ast
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

SEED = 42
DATA = "data/"   # put tmdb_5000_movies.csv and tmdb_5000_credits.csv here (from Kaggle)
OUT = "regression_out/"

os.makedirs(OUT, exist_ok=True)

# ---------- load ----------
movies = pd.read_csv(DATA + "tmdb_5000_movies.csv")
credits = pd.read_csv(DATA + "tmdb_5000_credits.csv")
df = movies.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id")

# ---------- features (same 27 predictors as the classification part) ----------
df["cast_size"] = df["cast"].apply(lambda s: len(ast.literal_eval(s)))
df["crew_size"] = df["crew"].apply(lambda s: len(ast.literal_eval(s)))
date = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = date.dt.year
df["release_month"] = date.dt.month
df["is_english"] = (df["original_language"] == "en").astype(int)
df["genre_list"] = df["genres"].apply(lambda s: [g["name"] for g in ast.literal_eval(s)])
genres = sorted({g for gl in df["genre_list"] for g in gl})
for g in genres:
    df["genre_" + g] = df["genre_list"].apply(lambda gl: int(g in gl))
df["budget"] = df["budget"].replace(0, np.nan)      # zero = missing
df["runtime"] = df["runtime"].replace(0, np.nan)

FEATURES = ["budget", "runtime", "cast_size", "crew_size", "release_year",
            "release_month", "is_english"] + ["genre_" + g for g in genres]


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def run(name, d, target, log_target):
    """70/30 split, train-only medians, LinearRegression + RandomForest + mean baseline."""
    X = d[FEATURES].copy()
    y = d[target].astype(float)
    if log_target:
        y = np.log10(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=SEED)
    med = Xtr.median()                       # medians from the training set only
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    # budget is heavily right-skewed: use log10 (all models see the same inputs)
    for part in (Xtr, Xte):
        part["budget"] = np.log10(part["budget"])
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            random_state=SEED, n_jobs=-1),
    }
    out = {"n_total": int(len(d)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
           "n_features": len(FEATURES), "models": {}}
    preds = pd.DataFrame({"target": name, "y_true": yte.values}, index=yte.index)
    for mname, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[mname] = p
        row = {"rmse": rmse(yte, p),
               "corr": float(np.corrcoef(yte, p)[0, 1]) if mname != "Mean baseline" else None,
               "mape": float(mean_absolute_percentage_error(yte, p))}
        if log_target:   # also on the original ratio scale
            row["rmse_ratio"] = rmse(10 ** yte, 10 ** p)
            row["mape_ratio"] = float(mean_absolute_percentage_error(10 ** yte, 10 ** p))
        out["models"][mname] = row
    return out, preds


# Target 1 (derived): log10 return on investment, ROI = revenue / budget.
# Films with budget or revenue below 10,000 USD are dropped (suspicious values).
d1 = df[(df["budget"] >= 10000) & (df["revenue"] >= 10000)].copy()
d1["roi"] = d1["revenue"] / d1["budget"]
r1, p1 = run("log10_roi", d1, "roi", True)

# Target 2 (derived): weighted rating (IMDb formula)
#   WR = v/(v+m) * R + m/(v+m) * C
# R = vote_average, v = vote_count, m = 20 (minimum votes, same as the classification
# filter), C = mean vote_average of the films with vote_count >= 20.
# It pulls ratings that rest on few votes toward the mean.
d2 = df[df["vote_count"] >= 20].copy()
M = 20
C = d2["vote_average"].mean()
d2["weighted_rating"] = (d2["vote_count"] / (d2["vote_count"] + M)) * d2["vote_average"] \
                        + (M / (d2["vote_count"] + M)) * C
r2, p2 = run("weighted_rating", d2, "weighted_rating", False)
r2["C_mean_rating"] = float(C)
r2["m_min_votes"] = M

res = {"genres": genres, "log10_roi": r1, "weighted_rating": r2}
json.dump(res, open(OUT + "regression_results.json", "w"), indent=2)
pd.concat([p1, p2]).to_csv(OUT + "regression_predictions.csv")
print(json.dumps(res, indent=2))
