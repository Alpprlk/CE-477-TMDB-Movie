import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

# Find dataset and figure paths relative to the project folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "tmdb_5000_movies.csv"
)

FIGURES_DIR = os.path.join(
    BASE_DIR,
    "figures"
)

os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

movies = pd.read_csv(DATA_PATH)

print("Dataset shape:", movies.shape)


# ---------------------------------------------------------
# 1. Runtime missing-value analysis
# ---------------------------------------------------------

missing_runtime = movies["runtime"].isna().sum()
zero_runtime = (movies["runtime"] == 0).sum()

print("\nRuntime missing-value analysis:")
print("Missing runtime values:", missing_runtime)
print("Zero runtime values:", zero_runtime)


# A zero-minute movie duration is not considered meaningful,
# so zero values are marked as missing in the cleaned column.
movies["runtime_clean"] = movies["runtime"].replace(
    0,
    np.nan
)

print(
    "Runtime values requiring treatment:",
    movies["runtime_clean"].isna().sum()
)


# ---------------------------------------------------------
# 2. Prepare budget and revenue
# ---------------------------------------------------------

# Zero budget and revenue values may represent missing or
# unreported values, so they are marked as NaN.
movies["budget_clean"] = movies["budget"].replace(
    0,
    np.nan
)

movies["revenue_clean"] = movies["revenue"].replace(
    0,
    np.nan
)

print("\nBudget and revenue cleaning:")
print(
    "Budget zero values treated as unknown:",
    movies["budget_clean"].isna().sum()
)

print(
    "Revenue zero values treated as unknown:",
    movies["revenue_clean"].isna().sum()
)


# ---------------------------------------------------------
# 3. IQR outlier detection
# ---------------------------------------------------------

def detect_iqr_outliers(series):
    """
    Detect suspected outliers using the 1.5 * IQR rule.
    """

    clean_series = series.dropna()

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (
        (series < lower_bound)
        |
        (series > upper_bound)
    )

    return (
        q1,
        q3,
        iqr,
        lower_bound,
        upper_bound,
        outliers
    )


columns_for_outlier_detection = [
    "budget_clean",
    "revenue_clean",
    "runtime_clean"
]


for column in columns_for_outlier_detection:

    (
        q1,
        q3,
        iqr,
        lower,
        upper,
        outlier_mask
    ) = detect_iqr_outliers(
        movies[column]
    )

    # Outliers are flagged rather than automatically removed.
    movies[column + "_outlier"] = outlier_mask

    print("\nOutlier analysis:", column)
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower bound:", lower)
    print("Upper bound:", upper)
    print("Number of outliers:", outlier_mask.sum())


# ---------------------------------------------------------
# 4. Runtime median imputation
# ---------------------------------------------------------

# Calculate the median using valid runtime observations only.
runtime_median = movies["runtime_clean"].median()

print("\nRuntime imputation:")
print("Runtime median:", runtime_median)


# Fill missing runtime values with the median.
movies["runtime_clean"] = movies["runtime_clean"].fillna(
    runtime_median
)

print(
    "Missing runtime values after imputation:",
    movies["runtime_clean"].isna().sum()
)


# ---------------------------------------------------------
# 5. Runtime discretization
# ---------------------------------------------------------

# Divide runtime values into three approximately equal-frequency
# ordinal categories.
movies["runtime_category"], runtime_bins = pd.qcut(
    movies["runtime_clean"],
    q=3,
    labels=[
        "Short",
        "Medium",
        "Long"
    ],
    retbins=True
)

runtime_counts = (
    movies["runtime_category"]
    .value_counts()
    .sort_index()
)

print("\nRuntime discretization:")
print("Bin boundaries:", runtime_bins)
print(runtime_counts)


# ---------------------------------------------------------
# 6. Runtime discretization visualization
# ---------------------------------------------------------

plt.figure(
    figsize=(6, 4)
)

runtime_counts.plot(
    kind="bar"
)

plt.title(
    "Movie Counts by Runtime Category"
)

plt.xlabel(
    "Runtime Category"
)

plt.ylabel(
    "Number of Movies"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()


figure_path = os.path.join(
    FIGURES_DIR,
    "runtime_discretization.png"
)

plt.savefig(
    figure_path,
    dpi=200
)

plt.close()


print("\nRuntime discretization figure saved to:")
print(figure_path)