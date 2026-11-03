import ast
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

print("movies:", movies.shape, "| credits:", credits.shape)
print("id/movie_id match:", set(movies["id"]) == set(credits["movie_id"]))
print(movies.dtypes)

# missing degerleri orijinal 20 sutun uzerinden hesapla (turetilmis sutunlar eklenmeden once)
miss = movies.isna().sum()

movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
movies["release_year"] = movies["release_date"].dt.year
movies["release_month"] = movies["release_date"].dt.month
print("release range:", movies["release_date"].min().date(), "->",
      movies["release_date"].max().date())

attr_types = pd.DataFrame({
    "Attribute": ["budget", "revenue", "runtime", "vote_count",
                  "popularity", "vote_average",
                  "release_year / release_month",
                  "original_language", "status",
                  "genres, keywords, production_*, spoken_languages",
                  "title, overview, tagline, homepage, id"],
    "Type / role": ["Numeric ratio", "Numeric ratio", "Numeric ratio",
                    "Numeric ratio", "Numeric score", "Numeric score",
                    "Interval / time (derived)", "Nominal", "Nominal",
                    "Multi-label JSON-like", "Free text / identifier"],
})
attr_types.to_csv("tables/attr_types.csv", index=False)

miss_pct = (miss / len(movies) * 100).round(2)
miss_df = pd.DataFrame({"Attribute": miss.index, "Missing": miss.values,
                        "Percent": miss_pct.values}).sort_values(
    "Missing", ascending=False)
miss_df = miss_df[miss_df["Missing"] > 0]
print(miss_df.to_string(index=False))
miss_df.to_csv("tables/missing.csv", index=False)

plt.figure(figsize=(6, 5))
m = miss[miss > 0].sort_values()
plt.barh(m.index, m.values)
plt.xlabel("Missing values")
plt.title("Attributes with Missing Values")
plt.tight_layout()
plt.savefig("figures/missing_values.png", dpi=200)
plt.close()

def parse(cell):
    try:
        return [d["name"] for d in ast.literal_eval(cell)]
    except (ValueError, SyntaxError, TypeError):
        return []

for col in ["genres", "keywords", "production_companies",
            "production_countries", "spoken_languages"]:
    movies[col + "_list"] = movies[col].apply(parse)
    print(col, "empty lists:", (movies[col + "_list"].str.len() == 0).sum())

for col in ["original_language", "status"]:
    vc = movies[col].value_counts()
    print(f"\n{col}: mode={vc.index[0]} count={vc.iloc[0]} "
          f"({vc.iloc[0] / len(movies) * 100:.2f}%), distinct={len(vc)}")
    print(vc.head(10))

gc = Counter(g for lst in movies["genres_list"] for g in lst)
rows = []
for col in ["original_language", "status"]:
    vc = movies[col].value_counts()
    rows.append((col, vc.index[0], vc.iloc[0],
                 round(vc.iloc[0] / len(movies) * 100, 2), len(vc)))
g, c = gc.most_common(1)[0]
rows.append(("genres", g, c, round(c / len(movies) * 100, 2), len(gc)))
mode_df = pd.DataFrame(rows, columns=["Attribute", "Mode", "Count", "Percent", "Distinct"])
print("\nMODE TABLE")
print(mode_df.to_string(index=False))
mode_df.to_csv("tables/mode_table.csv", index=False)

genre_counts = Counter(g for lst in movies["genres_list"] for g in lst)
print("\nTop genres:", genre_counts.most_common(10))
print("distinct genres:", len(genre_counts),
      "| avg per film:", movies["genres_list"].str.len().mean().round(2),
      "| max:", movies["genres_list"].str.len().max())

top_g = genre_counts.most_common(10)[::-1]
plt.figure(figsize=(7, 4))
plt.barh([g for g, _ in top_g], [c for _, c in top_g])
plt.xlabel("Number of movies")
plt.title("Ten Most Frequent Genres")
plt.tight_layout()
plt.savefig("figures/top_genres.png", dpi=200)
plt.close()

lang = movies["original_language"].value_counts().head(10)[::-1]
plt.figure(figsize=(7, 4))
plt.barh(lang.index, lang.values)
plt.xlabel("Number of movies")
plt.title("Most Frequent Original Languages")
plt.tight_layout()
plt.savefig("figures/top_languages.png", dpi=200)
plt.close()

status = movies["status"].value_counts()
plt.figure(figsize=(5, 3.5))
plt.bar(status.index, status.values)
plt.yscale("log")
plt.ylabel("Number of movies (log)")
plt.title("Movie Status")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("figures/status.png", dpi=200)
plt.close()

per_year = movies["release_year"].dropna().astype(int).value_counts().sort_index()
print("\nMost represented year:", per_year.idxmax(), per_year.max())
print("share of films >= 2000:",
      round((movies["release_year"] >= 2000).sum() /
            movies["release_year"].notna().sum() * 100, 1), "%")

plt.figure(figsize=(7, 3.5))
plt.plot(per_year.index, per_year.values)
plt.xlabel("Release year")
plt.ylabel("Number of movies")
plt.title("Movies per Release Year")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/movies_per_year.png", dpi=200)
plt.close()

top5 = [g for g, _ in genre_counts.most_common(5)]
rows = []
for _, r in movies.dropna(subset=["release_year"]).iterrows():
    dec = int(r["release_year"] // 10 * 10)
    for g in r["genres_list"]:
        if g in top5 and dec >= 1980:
            rows.append((dec, g))
dg = pd.DataFrame(rows, columns=["decade", "genre"])
pivot = dg.groupby(["decade", "genre"]).size().unstack(fill_value=0)[top5]
pivot.index = [f"{d}s" for d in pivot.index]
pivot.plot(kind="bar", stacked=True, figsize=(7, 4))
plt.xlabel("Release decade")
plt.ylabel("Genre assignments")
plt.title("Top Genres Across Recent Decades")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/genres_by_decade.png", dpi=200)
plt.close()

print("\nDone. Figures in ./figures, LaTeX tables in ./tables")
