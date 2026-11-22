import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# Cleaning: zero budget/runtime really mean "missing" in this dataset.
# Movies with very few votes have unreliable ratings, so we keep only those
# with at least 20 votes.
df = df[df["vote_count"] >= 20].copy()
for col in ["budget", "runtime"]:
    df[col] = df[col].replace(0, np.nan)

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

# Use movie attributes and genres as predictors for the rating class.
feature_cols = (
    ["budget", "runtime", "cast_size", "crew_size", "release_year", "release_month", "is_english"]
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

# Fill missing values in each split using medians from the training data.
impute_cols = ["budget", "runtime", "release_year", "release_month"]
train_medians = X_train[impute_cols].median()
X_train = X_train.fillna(train_medians)
X_val = X_val.fillna(train_medians)
X_test = X_test.fillna(train_medians)

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


# ======================================================================
# TASK 4: Model Visualization
# ======================================================================
plt.switch_backend("Agg")
figures_dir = BASE_DIR / "figures"
figures_dir.mkdir(exist_ok=True)

# The root is depth 0, so depth 2 shows the first three levels.
fig, ax = plt.subplots(figsize=(18, 8), layout="constrained")
annotations = plot_tree(
    dt, max_depth=2, feature_names=feature_cols,
    class_names=dt.classes_.tolist(), filled=True, rounded=True, ax=ax,
)
title = ax.set_title("Decision Tree (First Three Levels)")

# Remove the placeholder boxes for deeper branches and crop to the visible tree.
for annotation in annotations:
    if annotation.get_text().strip() == "(...)":
        annotation.remove()
fig.canvas.draw()
tree_bounds = Bbox.union([artist.get_window_extent() for artist in [title, *ax.texts]])
tree_bounds = tree_bounds.transformed(fig.dpi_scale_trans.inverted()).padded(0.15)
fig.savefig(figures_dir / "classification_decision_tree.png", dpi=180, bbox_inches=tree_bounds)
plt.close(fig)


# ======================================================================
# TASK 5: Classification Performance Comparison
# ======================================================================
models = {
    "Decision Tree": dt,
    "Random Forest": rf,
    "Logistic Regression": logreg,
    "KNN": knn,
}

for split_name, features, actual in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
    rows = []
    for model_name, model in models.items():
        predicted = model.predict(features)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, labels=class_labels, average="macro", zero_division=0
        )
        rows.append({
            "Model": model_name,
            "Accuracy": accuracy_score(actual, predicted),
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1-score": f1,
        })

        if split_name == "Test":
            fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
            ConfusionMatrixDisplay.from_predictions(
                actual, predicted, labels=class_labels, cmap="Blues", values_format="d", ax=ax
            )
            ax.set_title(f"{model_name} Test Confusion Matrix")
            filename = f"classification_confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(figures_dir / filename, dpi=180)
            plt.close(fig)

    results = pd.DataFrame(rows).set_index("Model")
    print("\n" + "=" * 76)
    print(f"{split_name.upper()} SET PERFORMANCE")
    print("=" * 76)
    print(results.to_string(float_format="{:.4f}".format))
    results.to_csv(BASE_DIR / f"classification_{split_name.lower()}_results.csv")

    if split_name == "Test":
        test_results = results

# Group the test scores by metric so the four models can be compared directly.
fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
test_results.T.plot.bar(ax=ax, rot=0, width=0.8)
for bars in ax.containers:
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
ax.set_title("Test Set Performance by Model")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend(title="Model", loc="upper center", ncol=4)
fig.savefig(figures_dir / "classification_model_comparison.png", dpi=180)
plt.close(fig)
