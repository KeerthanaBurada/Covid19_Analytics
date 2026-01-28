#!/usr/bin/env python3
"""
Experiment 1: COVID-19 Exploratory Data Analysis (EDA)

Dataset Columns:
Date | State | Region | Confirmed | Deaths | Recovered

Author: Keerthana Burada
Project: COVID-19 Analytics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
print("=" * 80)
print("COVID-19 DATA EXPLORATORY ANALYSIS")
print("=" * 80)

df = pd.read_csv("data/covid_19_data.csv")
print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# --------------------------------------------------
# 2. Standardize Column Names
# --------------------------------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

print("\nStandardized Columns:")
print(df.columns.tolist())

# --------------------------------------------------
# 3. Data Cleaning
# --------------------------------------------------
# Convert Date column
df["date"] = pd.to_datetime(df["date"])

# Fill missing state values
df["state"] = df["state"].fillna("Unknown")

# --------------------------------------------------
# 4. Basic Statistics
# --------------------------------------------------
print("\nBASIC STATISTICS")
print("-" * 80)
print(df[["confirmed", "deaths", "recovered"]].describe())

# --------------------------------------------------
# 5. Top 10 Regions by Confirmed Cases
# --------------------------------------------------
top_regions = (
    df.groupby("region")["confirmed"]
    .max()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Regions by Confirmed Cases:")
print(top_regions)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_regions.values, y=top_regions.index)
plt.title("Top 10 Regions by COVID-19 Confirmed Cases")
plt.xlabel("Confirmed Cases")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. Global Trend Over Time
# --------------------------------------------------
global_trend = (
    df.groupby("date")[["confirmed", "deaths", "recovered"]]
    .sum()
)

plt.figure(figsize=(12, 6))
plt.plot(global_trend.index, global_trend["confirmed"], label="Confirmed")
plt.plot(global_trend.index, global_trend["deaths"], label="Deaths")
plt.plot(global_trend.index, global_trend["recovered"], label="Recovered")

plt.title("Global COVID-19 Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Cases")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7. Death Rate Analysis
# --------------------------------------------------
df["death_rate"] = (df["deaths"] / df["confirmed"]) * 100

death_rate_region = (
    df.groupby("region")["death_rate"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Regions by Average Death Rate (%):")
print(death_rate_region.round(2))

# --------------------------------------------------
# 8. Key Insights
# --------------------------------------------------
print("\nKEY INSIGHTS")
print("-" * 80)
print("• COVID-19 cases increased steadily over time")
print("• Certain regions were disproportionately affected")
print("• Death rates varied significantly across regions")
print("• Data highlights importance of timely public health response")

print("\nEDA COMPLETED SUCCESSFULLY")
print("=" * 80)
