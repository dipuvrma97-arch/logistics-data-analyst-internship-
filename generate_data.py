"""
Week 2 - Step 1: Data Collection Simulation
Creates a SIMULATED QuickShip Logistics dataset (1000 orders, year 2025).
The column structure is inspired by public datasets such as
"DataCo Smart Supply Chain" and "Olist Brazilian E-Commerce" (Kaggle).

Some data-quality problems are added on purpose (missing values, duplicates,
outliers, wrong text format, invalid values) so that Week 2 cleaning has
something real to fix.

Run:  python week2/generate_data.py
Output: data/quickship_raw.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Works both as a script and when pasted into Google Colab
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)          # fixed seed -> same data every time
n = 1000

# ---------- 1. Generate clean base data ----------
order_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
warehouse = rng.choice(["Delhi", "Mumbai", "Kolkata", "Chennai"], n, p=[0.30, 0.30, 0.20, 0.20])
package_type = rng.choice(["Documents", "Electronics", "Clothing", "Grocery", "Furniture"],
                          n, p=[0.15, 0.25, 0.30, 0.15, 0.15])
distance_km = rng.integers(10, 900, n)
weight_kg = np.round(np.clip(rng.gamma(2.0, 6.0, n), 0.5, 80), 1)

# vehicle depends mostly on distance (Bike < 60 km, Van < 350 km, else Truck)
vehicle = np.where(distance_km < 60, "Bike", np.where(distance_km < 350, "Van", "Truck"))
r = rng.random(n)
vehicle = np.where((vehicle == "Bike") & (r < 0.30), "Van", vehicle)    # some short trips by Van
vehicle = np.where((vehicle == "Van") & (r >= 0.85), "Truck", vehicle)  # some medium trips by Truck

speed_km_per_day = pd.Series(vehicle).map({"Bike": 150, "Van": 300, "Truck": 400}).to_numpy()
warehouse_delay = pd.Series(warehouse).map(
    {"Delhi": 0.0, "Mumbai": 0.2, "Kolkata": 1.0, "Chennai": 0.3}).to_numpy()
festive_delay = np.isin(order_date.month, [10, 11]) * 0.6        # Oct-Nov festive rush
furniture_delay = (package_type == "Furniture") * 0.5

delivery_days = (0.5 + distance_km / speed_km_per_day + weight_kg / 60
                 + warehouse_delay + festive_delay + furniture_delay
                 + rng.normal(0, 0.4, n))
delivery_days = np.round(np.clip(delivery_days, 0.5, None), 1)

promised_days = np.ceil(distance_km / 300).astype(int) + 1

vehicle_extra = pd.Series(vehicle).map({"Bike": 0, "Van": 60, "Truck": 150}).to_numpy()
shipping_cost = 40 + 2.2 * distance_km + 6 * weight_kg + vehicle_extra + rng.normal(0, 60, n)
shipping_cost = np.round(np.clip(shipping_cost, 30, None), 0)

df = pd.DataFrame({
    "order_id": np.arange(1, n + 1),
    "order_date": order_date.strftime("%Y-%m-%d"),
    "warehouse": warehouse,
    "vehicle": vehicle,
    "package_type": package_type,
    "distance_km": distance_km,
    "weight_kg": weight_kg,
    "promised_days": promised_days,
    "delivery_days": delivery_days,
    "shipping_cost": shipping_cost,
})

# ---------- 2. Add data-quality problems on purpose ----------
perm = rng.permutation(n)                       # different rows for each problem
weight_missing = perm[:50]
days_missing = perm[50:70]
cost_outlier = perm[70:78]
days_typo = perm[78:82]
distance_bad = perm[82:85]
text_messy = perm[85:125]
dup_rows = perm[125:137]

df.loc[weight_missing, "weight_kg"] = np.nan                 # 50 missing weights
df.loc[days_missing, "delivery_days"] = np.nan               # 20 missing delivery days
df.loc[cost_outlier, "shipping_cost"] = (
    df.loc[cost_outlier, "shipping_cost"] * rng.integers(8, 12, len(cost_outlier)))  # 8 cost outliers
df.loc[days_typo, "delivery_days"] = [45, 50, 52, 60]        # 4 typing mistakes
df.loc[distance_bad, "distance_km"] = [-1, 0, -5]            # 3 invalid distances

wh = df["warehouse"].tolist()                                 # 40 messy warehouse names
for i in text_messy:
    k = rng.integers(0, 3)
    w = wh[i]
    wh[i] = w.lower() if k == 0 else (w.upper() if k == 1 else " " + w + " ")
df["warehouse"] = wh

df = pd.concat([df, df.loc[dup_rows]], ignore_index=True)    # 12 duplicate rows

df.to_csv(DATA / "quickship_raw.csv", index=False)
print("Raw dataset saved:", DATA / "quickship_raw.csv")
print("Shape:", df.shape)
print(df.head())
