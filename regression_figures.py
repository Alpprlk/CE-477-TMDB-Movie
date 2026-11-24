from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

BASE = Path(__file__).resolve().parent
pred = pd.read_csv(BASE / "regression_out" / "regression_predictions.csv", index_col=0)
(BASE / "figures").mkdir(exist_ok=True)
MODELS = ["Linear Regression", "Random Forest"]
TARGETS = {"log10_roi": r"$\log_{10}$ ROI", "weighted_rating": "Weighted rating"}

rows = []
for t, label in TARGETS.items():
    d = pred[pred["target"] == t]
    y = d["y_true"].values
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained", sharex=True, sharey=True)
    for ax, m in zip(axes, MODELS):
        p = d[m].values
        ax.scatter(p, y, s=10, alpha=0.35)
        lo, hi = min(p.min(), y.min()), max(p.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Ideal (y = x)")
        slope, icpt = np.polyfit(p, y, 1)           # regression line of actual on predicted
        xs = np.array([p.min(), p.max()])
        ax.plot(xs, slope * xs + icpt, "r-", lw=2, label="Regression line")
        r = np.corrcoef(y, p)[0, 1]
        ax.set_title(f"{m} (r = {r:.3f})")
        ax.set_xlabel(f"Predicted {label}")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel(f"Actual {label}")
    fig.suptitle(f"Test set: actual vs. predicted {label}")
    fig.savefig(BASE / "figures" / f"regression_scatter_{t}.png", dpi=180)
    plt.close(fig)
    for m in ["Mean baseline"] + MODELS:
        p = d[m].values
        row = {"Target": t, "Model": m,
               "RMSE": float(np.sqrt(mean_squared_error(y, p))),
               "MAPE": float(mean_absolute_percentage_error(y, p)),
               "Correlation": float(np.corrcoef(y, p)[0, 1]) if m != "Mean baseline" else np.nan}
        if t == "log10_roi":
            row["MAPE_ratio_scale"] = float(mean_absolute_percentage_error(10**y, 10**p))
        rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(BASE / "regression_out" / "regression_metrics.csv", index=False)
print(res.round(4).to_string(index=False))

# bar chart of RMSE relative to the mean baseline (comparison figure)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
for ax, metric in zip(axes, ["RMSE", "MAPE", "Correlation"]):
    piv = res.pivot(index="Target", columns="Model", values=metric)[["Mean baseline"] + MODELS]
    piv.rename(index={"log10_roi": "log10 ROI", "weighted_rating": "Weighted rating"}).plot.bar(ax=ax, rot=0, legend=False)
    ax.set_title(metric)
    for c in ax.containers: ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
axes[0].legend(fontsize=7)
fig.savefig(BASE / "figures" / "regression_model_comparison.png", dpi=180)
plt.close(fig)
