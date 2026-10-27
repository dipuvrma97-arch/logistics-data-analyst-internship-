"""
Week 3 - Advanced Data Analysis and Visualization in Logistics
Input : data/quickship_clean.csv   (made by week2/week2_cleaning.py)
Output: week3/charts/*.png (8 charts), week3/eda_summary.json

Run:  python week3/week3_eda.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
CHARTS = ROOT / "week3" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
df = pd.read_csv(ROOT / "data" / "quickship_clean.csv", parse_dates=["order_date"])
summary = {}

# ------------------------------------------------------------
# 1. EXPLORATORY DATA ANALYSIS (numbers)
# ------------------------------------------------------------
num_cols = ["distance_km", "weight_kg", "delivery_days", "shipping_cost"]

stats = df[num_cols].agg(["mean", "median", "std", "min", "max"]).T.round(2)
stats["skew"] = df[num_cols].skew().round(2)
summary["central_tendency"] = stats.reset_index().rename(columns={"index": "variable"}).to_dict("records")

corr = df[num_cols].corr().round(3)
summary["correlation"] = corr.to_dict()

summary["on_time_rate"] = round((df["is_late"] == 0).mean() * 100, 1)
summary["avg_delivery_days"] = round(df["delivery_days"].mean(), 2)
summary["avg_cost_per_km"] = round(df["shipping_cost"].sum() / df["distance_km"].sum(), 2)
summary["rows"] = int(len(df))

def group_table(by):
    g = df.groupby(by).agg(
        orders=("order_id", "count"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_delay_days=("delay_days", "mean"),
        on_time_pct=("is_late", lambda s: (1 - s.mean()) * 100),
        avg_cost=("shipping_cost", "mean"),
    ).round(2)
    return g.reset_index().to_dict("records")

summary["by_warehouse"] = group_table("warehouse")
summary["by_vehicle"] = group_table("vehicle")
summary["by_package"] = group_table("package_type")
summary["by_month"] = group_table("month")

print("=== Central tendency ===\n", stats)
print("\n=== Correlation ===\n", corr)
print("\nOn-time rate: ", summary["on_time_rate"], "%")
print("Average delivery days:", summary["avg_delivery_days"])
for key in ["by_warehouse", "by_vehicle", "by_package"]:
    print(f"\n=== {key} ===\n", pd.DataFrame(summary[key]))

# ------------------------------------------------------------
# 2. VISUALIZATIONS
# ------------------------------------------------------------
# Chart 1: Histogram - distribution of delivery time
plt.figure(figsize=(7, 4.5))
sns.histplot(df["delivery_days"], bins=25, kde=True, color="#2a7fba")
plt.axvline(df["delivery_days"].mean(), color="red", linestyle="--",
            label=f"Mean = {df['delivery_days'].mean():.2f} days")
plt.axvline(df["delivery_days"].median(), color="green", linestyle=":",
            label=f"Median = {df['delivery_days'].median():.2f} days")
plt.title("Chart 1: Distribution of delivery time")
plt.xlabel("Delivery time (days)")
plt.ylabel("Number of orders")
plt.legend()
plt.tight_layout()
plt.savefig(CHARTS / "01_delivery_time_histogram.png", dpi=150)
plt.close()

# Chart 2: Boxplot - delivery time by vehicle type
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x="vehicle", y="delivery_days", order=["Bike", "Van", "Truck"],
            hue="vehicle", palette="Set2", legend=False)
plt.title("Chart 2: Delivery time by vehicle type")
plt.xlabel("Vehicle type")
plt.ylabel("Delivery time (days)")
plt.tight_layout()
plt.savefig(CHARTS / "02_delivery_by_vehicle_boxplot.png", dpi=150)
plt.close()

# Chart 3: Scatter - distance vs shipping cost (with trend line)
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df, x="distance_km", y="shipping_cost", hue="vehicle",
                hue_order=["Bike", "Van", "Truck"], alpha=0.6)
z = np.polyfit(df["distance_km"], df["shipping_cost"], 1)
xs = np.linspace(df["distance_km"].min(), df["distance_km"].max(), 100)
plt.plot(xs, z[0] * xs + z[1], color="black", linewidth=1.5, label="Trend line")
plt.title("Chart 3: Distance vs shipping cost")
plt.xlabel("Distance (km)")
plt.ylabel("Shipping cost (Rs)")
plt.legend()
plt.tight_layout()
plt.savefig(CHARTS / "03_distance_vs_cost_scatter.png", dpi=150)
plt.close()

# Chart 4: Bar - on-time delivery % by warehouse
wh = pd.DataFrame(summary["by_warehouse"]).sort_values("on_time_pct")
plt.figure(figsize=(7, 4.5))
ax = sns.barplot(data=wh, x="warehouse", y="on_time_pct", hue="warehouse",
                 palette="viridis", legend=False)
for i, v in enumerate(wh["on_time_pct"]):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.axhline(90, color="red", linestyle="--", label="Target = 90%")
plt.ylim(0, 105)
plt.title("Chart 4: On-time delivery rate by warehouse")
plt.xlabel("Warehouse")
plt.ylabel("On-time deliveries (%)")
plt.legend()
plt.tight_layout()
plt.savefig(CHARTS / "04_ontime_by_warehouse_bar.png", dpi=150)
plt.close()

# Chart 5: Line - monthly average delivery time (trend / seasonality)
mon = df.groupby("month")["delivery_days"].mean().reset_index()
plt.figure(figsize=(7, 4.5))
plt.plot(mon["month"], mon["delivery_days"], marker="o", color="#d1495b")
plt.xticks(range(1, 13))
plt.title("Chart 5: Monthly average delivery time (2025)")
plt.xlabel("Month")
plt.ylabel("Average delivery time (days)")
plt.tight_layout()
plt.savefig(CHARTS / "05_monthly_trend_line.png", dpi=150)
plt.close()

# Chart 6: Heatmap - correlation between key variables
plt.figure(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Chart 6: Correlation heatmap")
plt.tight_layout()
plt.savefig(CHARTS / "06_correlation_heatmap.png", dpi=150)
plt.close()

# Chart 7: Bar - cost per km by vehicle type (cost driver)
veh_cost = (df.groupby("vehicle")["shipping_cost"].sum() / df.groupby("vehicle")["distance_km"].sum()).round(2)
veh_cost = veh_cost.reindex(["Bike", "Van", "Truck"])
summary["cost_per_km_by_vehicle"] = veh_cost.to_dict()
plt.figure(figsize=(7, 4.5))
ax = sns.barplot(x=veh_cost.index, y=veh_cost.values, hue=veh_cost.index, palette="Set2", legend=False)
for i, v in enumerate(veh_cost.values):
    ax.text(i, v + 0.1, f"Rs {v:.2f}", ha="center")
plt.title("Chart 7: Cost per km by vehicle type")
plt.xlabel("Vehicle type")
plt.ylabel("Cost per km (Rs)")
plt.tight_layout()
plt.savefig(CHARTS / "07_cost_per_km_by_vehicle_bar.png", dpi=150)
plt.close()

# Chart 8: Bar - on-time delivery % by package type
pk = pd.DataFrame(summary["by_package"]).sort_values("on_time_pct")
plt.figure(figsize=(7.5, 4.5))
ax = sns.barplot(data=pk, x="package_type", y="on_time_pct", hue="package_type",
                 palette="mako", legend=False)
for i, v in enumerate(pk["on_time_pct"]):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.axhline(90, color="red", linestyle="--", label="Target = 90%")
plt.ylim(0, 105)
plt.title("Chart 8: On-time delivery rate by package type")
plt.xlabel("Package type")
plt.ylabel("On-time deliveries (%)")
plt.legend()
plt.tight_layout()
plt.savefig(CHARTS / "08_ontime_by_package_bar.png", dpi=150)
plt.close()

with open(ROOT / "week3" / "eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved 8 charts in week3/charts/ and eda_summary.json")
