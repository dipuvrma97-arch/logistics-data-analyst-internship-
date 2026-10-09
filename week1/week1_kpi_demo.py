"""
Week 1 - Strategic Planning and Data Exploration in Logistics
Scenario : QuickShip Logistics (e-commerce courier company, India)
Purpose  : (1) Show how the 3 main KPIs are calculated with Python (pandas)
           (2) Show the pseudocode of the full analysis roadmap

Run:  python week1/week1_kpi_demo.py
"""

import pandas as pd

# ---------------------------------------------------------------
# PART 1: Tiny sample of 10 orders (hand-made) to illustrate KPIs
# ---------------------------------------------------------------
orders = pd.DataFrame({
    "order_id":      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "warehouse":     ["Delhi", "Delhi", "Mumbai", "Mumbai", "Kolkata",
                      "Kolkata", "Chennai", "Chennai", "Delhi", "Kolkata"],
    "distance_km":   [120, 450, 80, 600, 300, 700, 150, 520, 60, 250],
    "promised_days": [2, 3, 2, 3, 3, 3, 2, 3, 2, 3],
    "delivery_days": [1.8, 3.4, 1.5, 2.9, 3.6, 4.2, 2.1, 2.8, 1.2, 2.9],
    "shipping_cost": [330, 1180, 260, 1500, 840, 1900, 410, 1320, 190, 700],
})

# KPI 1: On-Time Delivery Rate (%) = on-time orders / total orders * 100
orders["is_on_time"] = orders["delivery_days"] <= orders["promised_days"]
on_time_rate = orders["is_on_time"].mean() * 100

# KPI 2: Average Delivery Time (days)
avg_delivery_days = orders["delivery_days"].mean()

# KPI 3: Cost per Kilometre (Rs/km) = total cost / total distance
cost_per_km = orders["shipping_cost"].sum() / orders["distance_km"].sum()

# KPI 4 (extra): average delay (delivery_days - promised_days) by warehouse
orders["delay_days"] = orders["delivery_days"] - orders["promised_days"]
delay_by_warehouse = orders.groupby("warehouse")["delay_days"].mean().round(2)

print("=== QuickShip KPI demo (10 sample orders) ===")
print(f"On-Time Delivery Rate : {on_time_rate:.1f} %")
print(f"Average Delivery Time : {avg_delivery_days:.2f} days")
print(f"Cost per Kilometre    : Rs {cost_per_km:.2f} / km")
print("\nAverage delay by warehouse (days; + means late):")
print(delay_by_warehouse)


# ---------------------------------------------------------------
# PART 2: Pseudocode of the end-to-end analysis roadmap
# ---------------------------------------------------------------
PSEUDOCODE = """
# STEP 1 - DATA COLLECTION
orders = load_data("quickship_raw.csv")            # public dataset / simulated

# STEP 2 - DATA CLEANING
remove duplicate rows
fix text formats (e.g. 'delhi ' -> 'Delhi')
fill missing weight with median of package type
remove invalid values (distance <= 0, delivery_days > 30)
remove extreme outliers using IQR rule (per vehicle type)
scale numeric columns with min-max normalization

# STEP 3 - EXPLORATORY DATA ANALYSIS (EDA)
calculate mean, median, std, correlation
plot histogram, boxplot, scatter, bar, line, heatmap
find bottlenecks (warehouse / vehicle / season)

# STEP 4 - PREDICTIVE MODELING
X = [distance, weight, vehicle, warehouse, package_type, month]
y = delivery_days
split data into train (80%) and test (20%)
train LinearRegression, DecisionTree, RandomForest
evaluate with RMSE, MAE, R-squared, 5-fold cross-validation
tune best model with GridSearchCV

# STEP 5 - OPTIMIZATION AND DECISIONS
what-if analysis: predicted delivery time if bottleneck is reduced
recommend vehicle allocation and warehouse capacity changes
"""

if __name__ == "__main__":
    print("\n=== Roadmap pseudocode ===")
    print(PSEUDOCODE)
