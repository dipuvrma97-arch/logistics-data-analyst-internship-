"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
Goal   : Predict delivery time (days) for a QuickShip order, then use the model
         for what-if scenarios to suggest ways to improve operations.
Input  : data/quickship_clean.csv   (made in Week 2)
Output : week4/model_results.json, week4/charts/*.png

Run:  python week4/week4_model.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
CHARTS = ROOT / "week4" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(ROOT / "data" / "quickship_clean.csv", parse_dates=["order_date"])
results = {}

# ------------------------------------------------------------
# 1. PREPARE DATA  (features X, target y)
# ------------------------------------------------------------
# In Week 3 only October and November were slow, so month (1-12) is turned into
# a festival-season flag: 1 = October or November, 0 = other months.
df["is_festive"] = df["month"].isin([10, 11]).astype(int)

feature_cols = ["distance_km", "weight_kg", "is_festive", "vehicle", "warehouse", "package_type"]
X = pd.get_dummies(df[feature_cols], columns=["vehicle", "warehouse", "package_type"])
y = df["delivery_days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
results["train_rows"] = int(len(X_train))
results["test_rows"] = int(len(X_test))
results["n_features"] = int(X.shape[1])

# ------------------------------------------------------------
# 2. TRAIN AND EVALUATE THREE MODELS
# ------------------------------------------------------------
def evaluate(model, name):
    model.fit(X_train, y_train)                                     # training
    pred = model.predict(X_test)                                    # testing
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")   # 5-fold validation
    return {
        "model": name,
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        "MAE": round(float(mean_absolute_error(y_test, pred)), 3),
        "R2": round(float(r2_score(y_test, pred)), 3),
        "CV_R2_mean": round(float(cv.mean()), 3),
        "CV_R2_std": round(float(cv.std()), 3),
    }

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree (max_depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest (default)": RandomForestRegressor(n_estimators=100, random_state=42),
}
table = [evaluate(m, name) for name, m in models.items()]

# ------------------------------------------------------------
# 3. HYPERPARAMETER TUNING (GridSearchCV on Random Forest)
# ------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 8, None],
    "min_samples_leaf": [1, 3],
}
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid,
                    cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
pred_best = best_rf.predict(X_test)
table.append({
    "model": "Random Forest (tuned)",
    "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred_best))), 3),
    "MAE": round(float(mean_absolute_error(y_test, pred_best)), 3),
    "R2": round(float(r2_score(y_test, pred_best)), 3),
    "CV_R2_mean": round(float(grid.best_score_), 3),
    "CV_R2_std": None,
})
results["model_table"] = table
results["best_params"] = grid.best_params_

# ------------------------------------------------------------
# 4. FINAL MODEL = Linear Regression (lowest test RMSE and easy to explain)
# ------------------------------------------------------------
lr = LinearRegression().fit(X_train, y_train)
pred_lr = lr.predict(X_test)

coef = pd.Series(lr.coef_, index=X.columns)
results["linear_coefs"] = {k: round(float(v), 3) for k, v in coef.items()}
results["linear_distance_per_100km"] = round(float(coef["distance_km"] * 100), 2)
results["linear_weight_per_10kg"] = round(float(coef["weight_kg"] * 10), 2)

# Feature importance from the tuned Random Forest
imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
results["feature_importance_top8"] = imp.head(8).round(3).to_dict()

# Chart 1: feature importance
plt.figure(figsize=(7.5, 4.8))
imp.head(8).sort_values().plot(kind="barh", color="#2a7fba")
plt.title("Top 8 factors affecting delivery time (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(CHARTS / "feature_importance.png", dpi=150)
plt.close()

# Chart 2: model comparison
comp = pd.DataFrame(table)
plt.figure(figsize=(7.5, 4.5))
ax = sns.barplot(data=comp, x="RMSE", y="model", hue="model", palette="crest", legend=False)
for i, v in enumerate(comp["RMSE"]):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center")
plt.title("Model comparison: RMSE on test data (lower is better)")
plt.xlabel("RMSE (days)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(CHARTS / "model_comparison.png", dpi=150)
plt.close()

# Chart 3: actual vs predicted
plt.figure(figsize=(6, 5.5))
plt.scatter(y_test, pred_lr, alpha=0.6, color="#d1495b")
lims = [min(y_test.min(), pred_lr.min()), max(y_test.max(), pred_lr.max())]
plt.plot(lims, lims, "k--", label="Perfect prediction")
plt.title("Actual vs predicted delivery time (test data)")
plt.xlabel("Actual delivery time (days)")
plt.ylabel("Predicted delivery time (days)")
plt.legend()
plt.tight_layout()
plt.savefig(CHARTS / "actual_vs_predicted.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 5. OPTIMIZATION: what-if scenarios on all orders (predicted by the linear model)
#    A = Kolkata works as fast as Delhi
#    B = festival rush is managed (Oct-Nov behave like normal months)
#    C = both A and B together
# ------------------------------------------------------------
promised = df["promised_days"].values
is_kol = (df["warehouse"] == "Kolkata").values
is_fest = (df["is_festive"] == 1).values

XA = X.copy()
XA.loc[is_kol, "warehouse_Kolkata"] = False
XA.loc[is_kol, "warehouse_Delhi"] = True
XB = X.copy()
XB.loc[is_fest, "is_festive"] = 0
XC = XA.copy()
XC.loc[is_fest, "is_festive"] = 0

pred_base = lr.predict(X)
scen = {
    "Current situation": pred_base,
    "A: Kolkata like Delhi": lr.predict(XA[X.columns]),
    "B: festival rush managed": lr.predict(XB[X.columns]),
    "C: both A and B": lr.predict(XC[X.columns]),
}
results["scenarios"] = [
    {"scenario": k,
     "avg_days": round(float(v.mean()), 2),
     "avg_saved": round(float((pred_base - v).mean()), 2),
     "ontime_pct": round(float((v <= promised).mean() * 100), 1)}
    for k, v in scen.items()]
results["scenario_kolkata_saved"] = round(float((pred_base - scen["A: Kolkata like Delhi"])[is_kol].mean()), 2)
results["scenario_festive_saved"] = round(float((pred_base - scen["B: festival rush managed"])[is_fest].mean()), 2)
results["kolkata_orders_all"] = int(is_kol.sum())
results["festive_orders_all"] = int(is_fest.sum())

# Chart 4: scenario comparison
sc = pd.DataFrame(results["scenarios"])
plt.figure(figsize=(7.5, 4.5))
ax = sns.barplot(data=sc, x="ontime_pct", y="scenario", hue="scenario", palette="crest", legend=False)
for i, v in enumerate(sc["ontime_pct"]):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
plt.xlim(0, 105)
plt.title("Predicted on-time share under each scenario")
plt.xlabel("Orders predicted to arrive within the promised days (%)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(CHARTS / "scenario_comparison.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 6. USING THE MODEL: predict delivery time of new orders at booking time
# ------------------------------------------------------------
def predict_one(distance_km, weight_kg, month, vehicle, warehouse, package_type):
    row = pd.DataFrame([{"distance_km": distance_km, "weight_kg": weight_kg,
                         "is_festive": int(month in (10, 11)), "vehicle": vehicle,
                         "warehouse": warehouse, "package_type": package_type}])
    row = pd.get_dummies(row, columns=["vehicle", "warehouse", "package_type"])
    row = row.reindex(columns=X.columns, fill_value=0)
    return float(lr.predict(row)[0])

examples = [
    ("Order A", 300, 10, 6, "Van", "Delhi", "Clothing"),
    ("Order B", 300, 10, 11, "Van", "Kolkata", "Furniture"),
    ("Order C", 700, 20, 3, "Truck", "Mumbai", "Electronics"),
]
results["example_predictions"] = [
    {"order": e[0], "distance_km": e[1], "weight_kg": e[2], "month": e[3], "vehicle": e[4],
     "warehouse": e[5], "package_type": e[6],
     "promised_days": math.ceil(e[1] / 300) + 1,
     "predicted_days": round(predict_one(*e[1:]), 2)} for e in examples]

# ------------------------------------------------------------
# 7. EXTRA SUMMARY NUMBERS for the report
# ------------------------------------------------------------
veh = df.groupby("vehicle").agg(
    orders=("order_id", "count"),
    total_cost=("shipping_cost", "sum"),
    total_km=("distance_km", "sum"),
    on_time_pct=("is_late", lambda s: (1 - s.mean()) * 100),
).round(2)
veh["cost_per_km"] = (veh["total_cost"] / veh["total_km"]).round(2)
results["vehicle_summary"] = veh[["orders", "cost_per_km", "on_time_pct"]].reset_index().to_dict("records")

results["festive_avg_days"] = round(float(df.loc[is_fest, "delivery_days"].mean()), 2)
results["normal_avg_days"] = round(float(df.loc[~is_fest, "delivery_days"].mean()), 2)
results["festive_ontime_pct"] = round(float((1 - df.loc[is_fest, "is_late"].mean()) * 100), 1)
results["normal_ontime_pct"] = round(float((1 - df.loc[~is_fest, "is_late"].mean()) * 100), 1)

with open(ROOT / "week4" / "model_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# ------------------------------------------------------------
# Print the main results
# ------------------------------------------------------------
print("=== Model results on test data ===")
print(pd.DataFrame(table).fillna("-").to_string(index=False))
print("\nBest parameters (Random Forest):", grid.best_params_)
print("\n=== Top factors (Random Forest importance) ===")
print(imp.head(8).round(3).to_string())
print("\n=== What-if scenarios (predicted, all orders) ===")
print(sc.to_string(index=False))
print("\n=== Example predictions for new orders ===")
print(pd.DataFrame(results["example_predictions"])[["order", "distance_km", "weight_kg", "month",
      "vehicle", "warehouse", "package_type", "promised_days", "predicted_days"]].to_string(index=False))
