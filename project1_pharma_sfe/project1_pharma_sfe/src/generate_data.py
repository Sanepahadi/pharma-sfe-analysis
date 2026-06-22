"""
Synthetic Pharma Sales Dataset Generator
Simulates realistic sales rep performance data across territories and drug categories.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

DRUGS = ["CardioMax", "DiabeCure", "NeuroShield", "OncoPrime", "RespiClear"]
REGIONS = ["North", "South", "East", "West", "Central"]
REPS = [f"REP_{i:03d}" for i in range(1, 51)]

REP_REGION = {rep: random.choice(REGIONS) for rep in REPS}
REP_EXPERIENCE = {rep: random.randint(1, 15) for rep in REPS}

DRUG_BASE_SALES = {
    "CardioMax": 120000,
    "DiabeCure": 95000,
    "NeuroShield": 75000,
    "OncoPrime": 200000,
    "RespiClear": 60000,
}

DRUG_QUOTA = {
    "CardioMax": 130000,
    "DiabeCure": 100000,
    "NeuroShield": 80000,
    "OncoPrime": 210000,
    "RespiClear": 65000,
}

SEASONALITY = {
    1: 0.85, 2: 0.88, 3: 0.95, 4: 1.02,
    5: 1.05, 6: 1.00, 7: 0.92, 8: 0.90,
    9: 1.05, 10: 1.10, 11: 1.08, 12: 1.15
}

REGION_MULTIPLIER = {
    "North": 1.05, "South": 0.95, "East": 1.10,
    "West": 1.00, "Central": 0.90
}

records = []
start_date = datetime(2022, 1, 1)

for rep in REPS:
    region = REP_REGION[rep]
    exp = REP_EXPERIENCE[rep]
    skill_factor = 0.75 + (exp / 15) * 0.50 + np.random.normal(0, 0.08)
    skill_factor = max(0.5, min(1.5, skill_factor))

    for drug in DRUGS:
        for month_offset in range(24):
            date = start_date + timedelta(days=30 * month_offset)
            month = date.month
            year = date.year

            base = DRUG_BASE_SALES[drug]
            quota = DRUG_QUOTA[drug]
            region_mult = REGION_MULTIPLIER[region]
            season_mult = SEASONALITY[month]
            noise = np.random.normal(1.0, 0.07)

            actual_sales = base * skill_factor * region_mult * season_mult * noise
            calls_made = int(np.random.normal(45, 8) * (skill_factor * 0.6 + 0.4))
            calls_made = max(15, calls_made)
            conversion_rate = round(np.random.beta(2 + skill_factor, 5) * 100, 2)

            records.append({
                "rep_id": rep,
                "region": region,
                "drug": drug,
                "year": year,
                "month": month,
                "date": date.strftime("%Y-%m-%d"),
                "actual_sales": round(actual_sales, 2),
                "quota": quota,
                "quota_attainment_pct": round((actual_sales / quota) * 100, 2),
                "calls_made": calls_made,
                "conversion_rate_pct": conversion_rate,
                "rep_experience_years": exp,
            })

df = pd.DataFrame(records)
df.to_csv("/home/claude/projects/project1_pharma_sfe/data/pharma_sales_data.csv", index=False)
print(f"Dataset created: {len(df)} rows, {df['rep_id'].nunique()} reps, {df['drug'].nunique()} drugs")
print(df.head())
