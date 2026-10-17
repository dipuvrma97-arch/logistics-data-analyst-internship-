"""
Week 2 - Data Collection, Cleaning and Preprocessing for Logistics Analysis
Input : data/quickship_raw.csv   (made by week2/generate_data.py)
Output: data/quickship_clean.csv, week2/cleaning_summary.json,
        week2/charts/cost_outliers_before_after.png

Run:  python week2/week2_cleaning.py
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
(ROOT / "week2" / "charts").mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(ROOT / "data" / "quickship_raw.csv")
df = raw.copy()
log = {}

# ------------------------------------------------------------
# STEP 0: Inspect the raw data
# ------------------------------------------------------------
log["raw_rows"] = int(len(raw))
log["raw_cols"] = int(raw.shape[1])
log["missing_before"] = {c: int(v) for c, v in raw.isnull().sum().items() if v > 0}
log["duplicates_before"] = int(raw.duplicated().sum())
log["warehouse_raw_unique"] = int(raw["warehouse"].nunique())
log["invalid_distance"] = int((raw["distance_km"] <= 0).sum())
log["delivery_days_over_30"] = int((raw["delivery_days"] > 30).sum())

print("Raw shape:", raw.shape)
print("\nMissing values:\n", raw.isnull().sum())
print("\nDuplicate rows:", log["duplicates_before"])
print("\nWarehouse names (raw):", sorted(raw["warehouse"].unique()))

# ------------------------------------------------------------
# STEP 1: Fix wrong text format  ('delhi ' -> 'Delhi') and date type
# ------------------------------------------------------------
df["warehouse"] = df["warehouse"].str.strip().str.title()
log["warehouse_rows_fixed"] = int((raw["warehouse"] != df["warehouse"]).sum())
log["warehouse_clean_unique"] = int(df["warehouse"].nunique())
df["order_date"] = pd.to_datetime(df["order_date"])

# ------------------------------------------------------------
# STEP 2: Remove duplicate rows
# ------------------------------------------------------------
n0 = len(df)
df = df.drop_duplicates()
log["duplicates_removed"] = int(n0 - len(df))

# ------------------------------------------------------------
# STEP 3: Remove invalid values (domain rules)
# ------------------------------------------------------------
n0 = len(df)
df = df[df["distance_km"] > 0]
log["invalid_distance_removed"] = int(n0 - len(df))

n0 = len(df)
df = df[df["delivery_days"].notna()]          # target column must not be missing
log["missing_days_removed"] = int(n0 - len(df))

n0 = len(df)
df = df[df["delivery_days"] <= 30]            # >30 days is a typing mistake
log["days_typo_removed"] = int(n0 - len(df))

# ------------------------------------------------------------
# STEP 4: Handle missing values  (weight -> median of same package type)
# ------------------------------------------------------------
log["weight_missing_filled"] = int(df["weight_kg"].isnull().sum())
df["weight_kg"] = df["weight_kg"].fillna(
    df.groupby("package_type")["weight_kg"].transform("median"))
df["weight_kg"] = df["weight_kg"].round(1)

# ------------------------------------------------------------
# STEP 5: Outlier detection with IQR rule (per vehicle type)
#         far-out fence = 3 x IQR so genuine heavy/long shipments are kept
# ------------------------------------------------------------
before_outlier = df.copy()
q1 = df.groupby("vehicle")["shipping_cost"].transform(lambda s: s.quantile(0.25))
q3 = df.groupby("vehicle")["shipping_cost"].transform(lambda s: s.quantile(0.75))
iqr = q3 - q1
is_outlier = (df["shipping_cost"] < q1 - 3 * iqr) | (df["shipping_cost"] > q3 + 3 * iqr)
log["cost_outliers_removed"] = int(is_outlier.sum())
log["max_cost_before"] = float(before_outlier["shipping_cost"].max())
log["mean_cost_before"] = round(float(before_outlier["shipping_cost"].mean()), 1)
df = df[~is_outlier]
log["max_cost_after"] = float(df["shipping_cost"].max())
log["mean_cost_after"] = round(float(df["shipping_cost"].mean()), 1)
log["cost_per_km_before"] = round(float(before_outlier["shipping_cost"].sum() / before_outlier["distance_km"].sum()), 2)
log["cost_per_km_after"] = round(float(df["shipping_cost"].sum() / df["distance_km"].sum()), 2)

# ------------------------------------------------------------
# STEP 6: Create useful new columns (feature engineering)
# ------------------------------------------------------------
df["month"] = df["order_date"].dt.month
df["delay_days"] = (df["delivery_days"] - df["promised_days"]).round(1)
df["is_late"] = (df["delay_days"] > 0).astype(int)
df["cost_per_km"] = (df["shipping_cost"] / df["distance_km"]).round(2)

# ------------------------------------------------------------
# STEP 7: Normalization (Min-Max scaling to 0-1)
#         x_norm = (x - min) / (max - min)
# ------------------------------------------------------------
for col in ["distance_km", "weight_kg", "shipping_cost"]:
    df[col + "_norm"] = ((df[col] - df[col].min()) / (df[col].max() - df[col].min())).round(4)

# ------------------------------------------------------------
# STEP 8: Final check and save
# ------------------------------------------------------------
df = df.reset_index(drop=True)
log["clean_rows"] = int(len(df))
log["rows_removed_total"] = int(len(raw) - len(df))
log["missing_after"] = int(df.isnull().sum().sum())
log["duplicates_after"] = int(df.duplicated().sum())
log["norm_min_max_check"] = {c: [float(df[c].min()), float(df[c].max())]
                             for c in ["distance_km_norm", "weight_kg_norm", "shipping_cost_norm"]}

df.to_csv(ROOT / "data" / "quickship_clean.csv", index=False)
with open(ROOT / "week2" / "cleaning_summary.json", "w") as f:
    json.dump(log, f, indent=2)

# ------------------------------------------------------------
# Chart: shipping cost before vs after outlier removal
# ------------------------------------------------------------
order = ["Bike", "Van", "Truck"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(data=before_outlier, x="vehicle", y="shipping_cost", order=order,
            hue="vehicle", palette="Set2", legend=False, ax=axes[0])
axes[0].set_title("Before outlier removal")
sns.boxplot(data=df, x="vehicle", y="shipping_cost", order=order,
            hue="vehicle", palette="Set2", legend=False, ax=axes[1])
axes[1].set_title("After outlier removal")
for ax in axes:
    ax.set_xlabel("Vehicle type")
    ax.set_ylabel("Shipping cost (Rs)")
plt.suptitle("Shipping cost by vehicle type", fontsize=13)
plt.tight_layout()
plt.savefig(ROOT / "week2" / "charts" / "cost_outliers_before_after.png", dpi=150)
plt.close()


# ------------------------------------------------------------
# Chart: how many problems of each type were found in the raw data
# ------------------------------------------------------------
issues = {
    "Missing weight": log["missing_before"]["weight_kg"],
    "Wrong warehouse spelling": log["warehouse_rows_fixed"],
    "Missing delivery days": log["missing_before"]["delivery_days"],
    "Duplicate rows": log["duplicates_before"],
    "Extreme shipping cost": log["cost_outliers_removed"],
    "Impossible delivery days (>30)": log["delivery_days_over_30"],
    "Invalid distance (<= 0)": log["invalid_distance"],
}
plt.figure(figsize=(8, 4.5))
ax = sns.barplot(x=list(issues.values()), y=list(issues.keys()), hue=list(issues.keys()),
                 palette="flare", legend=False)
for i, v in enumerate(issues.values()):
    ax.text(v + 0.5, i, str(v), va="center")
plt.title("Data quality problems found in the raw data")
plt.xlabel("Number of rows")
plt.tight_layout()
plt.savefig(ROOT / "week2" / "charts" / "data_quality_issues.png", dpi=150)
plt.close()

print("\n=== CLEANING SUMMARY ===")
print(json.dumps(log, indent=2))
print("\nClean data saved to data/quickship_clean.csv  shape:", df.shape)
