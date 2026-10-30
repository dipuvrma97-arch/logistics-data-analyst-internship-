# Logistics Data Analyst Internship - QuickShip Logistics

Four-week project done for the **Logistics Data Analyst Intern** role at YuvaIntern (06-Oct-2026 to 03-Nov-2026).
Author: **Deepu Verma**

**Case study:** QuickShip Logistics, a fictional e-commerce courier company with four warehouses (Delhi, Mumbai, Kolkata, Chennai).
Problem: late deliveries and rising shipping cost. The dataset is **simulated** in Python (structure inspired by the public
DataCo Smart Supply Chain and Olist Brazilian E-Commerce datasets); it is not real company data.

| Week | Topic | Code | Report |
|------|-------|------|--------|
| 1 | Strategic planning and data exploration | `week1/week1_kpi_demo.py` | `week1/Week1_Strategic_Planning_QuickShip.docx` |
| 2 | Data collection, cleaning, preprocessing | `week2/generate_data.py`, `week2/week2_cleaning.py` | `week2/Week2_Data_Cleaning_QuickShip.docx` |
| 3 | Analysis and visualization (EDA) | `week3/week3_eda.py` | `week3/Week3_EDA_Visualization_QuickShip.docx` |
| 4 | Predictive modeling and optimization | `week4/week4_model.py` | `week4/Week4_Predictive_Modeling_QuickShip.docx` |

## How to run
```bash
pip install -r requirements.txt
python week1/week1_kpi_demo.py
python week2/generate_data.py      # creates data/quickship_raw.csv
python week2/week2_cleaning.py     # creates data/quickship_clean.csv
python week3/week3_eda.py          # creates charts in week3/charts
python week4/week4_model.py        # trains models, creates charts in week4/charts
```
Run the files in this order. Everything also works in Google Colab (paste each file in a cell, in the same order).

## Main findings (simulated data)
- Overall on-time delivery rate is 82.5% against a 90% target.
- Distance is the main driver of both delivery time and cost.
- The Kolkata warehouse is the biggest bottleneck; October-November (festive season) is the seasonal peak.
- Linear Regression predicted delivery time best (R2 about 0.78, MAE about 0.34 days).
