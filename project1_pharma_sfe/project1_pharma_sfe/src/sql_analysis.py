"""
SQL-Based Sales Analysis using SQLite
Demonstrates ability to write complex analytical SQL queries
"""

import sqlite3
import pandas as pd

# Load CSV into SQLite (simulates a real database)
df = pd.read_csv("/home/claude/projects/project1_pharma_sfe/data/pharma_sales_data.csv")
conn = sqlite3.connect(":memory:")
df.to_sql("pharma_sales", conn, index=False, if_exists="replace")
print("✓ Database loaded\n")

queries = {

"1. Top 10 Reps by Total Sales": """
    SELECT
        rep_id,
        region,
        ROUND(SUM(actual_sales), 0)        AS total_sales,
        ROUND(AVG(quota_attainment_pct), 2) AS avg_quota_attainment_pct,
        ROUND(AVG(calls_made), 1)           AS avg_monthly_calls
    FROM pharma_sales
    GROUP BY rep_id, region
    ORDER BY total_sales DESC
    LIMIT 10
""",

"2. Quarterly Sales Trend by Drug": """
    SELECT
        year,
        CASE
            WHEN month BETWEEN 1 AND 3  THEN 'Q1'
            WHEN month BETWEEN 4 AND 6  THEN 'Q2'
            WHEN month BETWEEN 7 AND 9  THEN 'Q3'
            ELSE 'Q4'
        END                                     AS quarter,
        drug,
        ROUND(SUM(actual_sales), 0)             AS total_sales,
        ROUND(AVG(quota_attainment_pct), 2)     AS avg_attainment_pct
    FROM pharma_sales
    GROUP BY year, quarter, drug
    ORDER BY year, quarter, total_sales DESC
    LIMIT 20
""",

"3. Region Performance vs Quota": """
    SELECT
        region,
        ROUND(SUM(actual_sales), 0)             AS total_sales,
        ROUND(AVG(quota_attainment_pct), 2)     AS avg_attainment_pct,
        COUNT(DISTINCT rep_id)                  AS num_reps,
        ROUND(AVG(calls_made), 1)               AS avg_calls_per_rep,
        ROUND(AVG(conversion_rate_pct), 2)      AS avg_conversion_pct,
        CASE
            WHEN AVG(quota_attainment_pct) >= 100 THEN 'On/Above Target'
            WHEN AVG(quota_attainment_pct) >= 90  THEN 'Near Target'
            ELSE 'Below Target'
        END                                     AS performance_band
    FROM pharma_sales
    GROUP BY region
    ORDER BY avg_attainment_pct DESC
""",

"4. Underperforming Reps (Below 90% Quota)": """
    SELECT
        rep_id,
        region,
        rep_experience_years,
        ROUND(AVG(quota_attainment_pct), 2)  AS avg_attainment_pct,
        ROUND(AVG(calls_made), 1)            AS avg_calls,
        ROUND(AVG(conversion_rate_pct), 2)   AS avg_conversion_pct,
        COUNT(*) AS months_tracked
    FROM pharma_sales
    GROUP BY rep_id, region, rep_experience_years
    HAVING avg_attainment_pct < 90
    ORDER BY avg_attainment_pct ASC
    LIMIT 15
""",

"5. Drug Revenue Ranking with Running Total": """
    SELECT
        drug,
        ROUND(SUM(actual_sales), 0)  AS total_sales,
        ROUND(AVG(quota_attainment_pct), 2) AS avg_attainment_pct,
        ROUND(
            SUM(SUM(actual_sales)) OVER (ORDER BY SUM(actual_sales) DESC)
            / SUM(SUM(actual_sales)) OVER () * 100, 2
        )                            AS cumulative_pct_of_revenue
    FROM pharma_sales
    GROUP BY drug
    ORDER BY total_sales DESC
""",

"6. Month-over-Month Growth Rate": """
    WITH monthly AS (
        SELECT
            year, month,
            SUM(actual_sales) AS monthly_sales
        FROM pharma_sales
        GROUP BY year, month
    ),
    with_prev AS (
        SELECT *,
            LAG(monthly_sales) OVER (ORDER BY year, month) AS prev_month_sales
        FROM monthly
    )
    SELECT
        year, month, monthly_sales,
        ROUND((monthly_sales - prev_month_sales) / prev_month_sales * 100, 2)
            AS mom_growth_pct
    FROM with_prev
    WHERE prev_month_sales IS NOT NULL
    ORDER BY year, month
    LIMIT 12
""",
}

for title, query in queries.items():
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    result = pd.read_sql_query(query, conn)
    print(result.to_string(index=False))

conn.close()
print("\n✅ SQL analysis complete")
