import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent


def find_file(name):
    """Look next to the script, then in 'data', then in any subfolder."""
    for folder in (BASE_DIR, BASE_DIR / "data"):
        if (folder / name).exists():
            return folder / name
    matches = list(BASE_DIR.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find {name} in {BASE_DIR} or any folder inside it."
    )


def parse_names(cell, key="name"):
    """Turn a JSON-like string column into a list of names."""
    try:
        return [d[key] for d in ast.literal_eval(cell)]
    except (ValueError, SyntaxError):
        return []


# ======================================================================
# DATA LOADING AND PREPARATION (merge the two files, build features)
# ======================================================================
movies = pd.read_csv(find_file("tmdb_5000_movies.csv"))
credits = pd.read_csv(find_file("tmdb_5000_credits.csv"))

# The credits file links to movies through movie_id == id
credits = credits.drop(columns=["title"]).rename(columns={"movie_id": "id"})
df = movies.merge(credits, on="id", how="inner")


# Genres -> one-hot columns for the most common genres
df["genre_list"] = df["genres"].apply(parse_names)
top_genres = pd.Series([g for gl in df["genre_list"] for g in gl]).value_counts().index.tolist()
for g in top_genres:
    df[f"genre_{g}"] = df["genre_list"].apply(lambda gl, g=g: int(g in gl))

# Simple numeric features from other columns
df["cast_size"] = df["cast"].apply(lambda c: len(parse_names(c)))
df["crew_size"] = df["crew"].apply(lambda c: len(parse_names(c)))
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year
df["release_month"] = df["release_date"].dt.month
df["is_english"] = (df["original_language"] == "en").astype(int)

# Cleaning: zero budget/revenue/runtime really mean "missing" in this dataset.
# Movies with very few votes have unreliable ratings, so we keep only those
# with at least 20 votes.
df = df[df["vote_count"] >= 20].copy()
for col in ["budget", "revenue", "runtime"]:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())
df["release_year"] = df["release_year"].fillna(df["release_year"].median())
df["release_month"] = df["release_month"].fillna(df["release_month"].median())

print("Rows after cleaning:", len(df))


# ======================================================================
# TASK 1: Select a discrete target attribute
#         Vote average is continuous thus it needs to be discretized...
# ======================================================================
# 'vote_average' is a number between 0 and 10, so it is not a class label yet.
# We turn it into 3 classes with QUANTILE (equal-frequency) discretization:
# the cut points are chosen so that each class gets about 1/3 of the movies.
# This keeps the classes balanced, unlike fixed ranges such as 0-4 / 4-7 / 7-10
# where almost every movie would fall into the middle class.
class_labels = ["Low", "Medium", "High"]

# retbins=True also returns the cut points (bin edges) that qcut calculated
df["rating_class"], bin_edges = pd.qcut(
    df["vote_average"], q=3, labels=class_labels, retbins=True
)

# --- Show exactly how the discretization was done ---
print("\n" + "=" * 60)
print("DISCRETIZATION OF vote_average (quantile / equal-frequency, 3 bins)")
print("=" * 60)

# The cut points: the 0%, 33.3%, 66.7% and 100% quantiles of vote_average
print("Bin edges (cut points):", [round(float(e), 2) for e in bin_edges])
print("Rule (each bin is open on the left and closed on the right,")
print("      except the first bin, which also includes its lowest value):")
print(f"  Low    : {bin_edges[0]:.1f} <= vote_average <= {bin_edges[1]:.1f}")
print(f"  Medium : {bin_edges[1]:.1f} <  vote_average <= {bin_edges[2]:.1f}")
print(f"  High   : {bin_edges[2]:.1f} <  vote_average <= {bin_edges[3]:.1f}")

# (The classes are only roughly 1/3 each because many movies share the same
#  rating, e.g. 6.0, so a cut point cannot split them apart.)
# How many movies landed in each class, and the real min/max inside each class
summary = (
    df.groupby("rating_class", observed=True)["vote_average"]
      .agg(movies="count", lowest="min", highest="max")
      .reindex(class_labels)
)
summary["share"] = (summary["movies"] / summary["movies"].sum() * 100).round(1).astype(str) + "%"
print("\nResulting classes (actual values found in the data):")
print(summary.to_string())
print("=" * 60)

# Features. vote_average is excluded because the target is derived from it.
feature_cols = (
    ["budget", "revenue", "runtime", "popularity", "vote_count",
     "cast_size", "crew_size", "release_year", "release_month", "is_english"]
    + [f"genre_{g}" for g in top_genres]
)
X = df[feature_cols]
y = df["rating_class"].astype(str)


# ======================================================================
# TASK 2: Split into training (60%), validation (20%), testing (20%)
# ======================================================================
# First hold out 20% for testing, then take 25% of the remaining 80%
# (= 20% of the total) for validation. Stratify to keep class proportions.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
)

print(f"\nTrain: {len(X_train)} ({len(X_train)/len(X):.0%})")
print(f"Validation: {len(X_val)} ({len(X_val)/len(X):.0%})")
print(f"Test: {len(X_test)} ({len(X_test)/len(X):.0%})")


# ======================================================================
# TASK 3, ALGORITHM 1: Decision Tree
# ======================================================================
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)


# ======================================================================
# TASK 3, ALGORITHM 2: Random Forest
# ======================================================================
rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                            n_jobs=-1, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)


# ======================================================================
# TASK 3, ALGORITHM 3: Logistic Regression (with feature scaling)
# ======================================================================
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])
logreg.fit(X_train, y_train)


# ======================================================================
# TASK 3, ALGORITHM 4: K-Nearest Neighbors (with feature scaling)
# ======================================================================
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=15)),
])
knn.fit(X_train, y_train)

print("\nAll 4 models trained: Decision Tree, Random Forest, Logistic Regression, KNN")