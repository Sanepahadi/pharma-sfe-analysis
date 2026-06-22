"""
Pharma Sales Force Effectiveness Analysis
==========================================
Full EDA + XGBoost Forecasting + Business Insights
Author: [Your Name]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "font.size": 11,
})
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
OUT = "/home/claude/projects/project1_pharma_sfe/outputs/"

# ── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/projects/project1_pharma_sfe/data/pharma_sales_data.csv")
df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df):,} records | {df['rep_id'].nunique()} reps | {df['drug'].nunique()} drugs")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Executive KPI Dashboard
# ═══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Pharma Sales Force Effectiveness — Executive KPI Dashboard",
             fontsize=18, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a. Monthly total sales trend
ax1 = fig.add_subplot(gs[0, :2])
monthly = df.groupby(["year", "month"])["actual_sales"].sum().reset_index()
monthly["period"] = pd.to_datetime(
    monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
)
monthly = monthly.sort_values("period")
ax1.plot(monthly["period"], monthly["actual_sales"] / 1e6,
         color=PALETTE[0], linewidth=2.5, marker="o", markersize=4)
ax1.fill_between(monthly["period"], monthly["actual_sales"] / 1e6,
                 alpha=0.15, color=PALETTE[0])
ax1.set_title("Total Monthly Sales (All Drugs, All Reps)", fontweight="bold")
ax1.set_ylabel("Sales ($ Millions)")
ax1.set_xlabel("")
ax1.tick_params(axis="x", rotation=30)

# 1b. Sales by Drug
ax2 = fig.add_subplot(gs[0, 2])
drug_sales = df.groupby("drug")["actual_sales"].sum().sort_values(ascending=True)
bars = ax2.barh(drug_sales.index, drug_sales.values / 1e6, color=PALETTE)
ax2.set_title("Total Sales by Drug", fontweight="bold")
ax2.set_xlabel("Sales ($ Millions)")
for bar, val in zip(bars, drug_sales.values / 1e6):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
             f"${val:.1f}M", va="center", fontsize=9)

# 1c. Quota Attainment by Region
ax3 = fig.add_subplot(gs[1, 0])
region_qa = df.groupby("region")["quota_attainment_pct"].mean().sort_values(ascending=False)
colors = [PALETTE[2] if v >= 100 else PALETTE[3] for v in region_qa.values]
bars3 = ax3.bar(region_qa.index, region_qa.values, color=colors, edgecolor="white", linewidth=1.5)
ax3.axhline(100, color="black", linestyle="--", linewidth=1.5, label="Quota = 100%")
ax3.set_title("Avg Quota Attainment by Region", fontweight="bold")
ax3.set_ylabel("Quota Attainment %")
ax3.legend(fontsize=9)
for bar, val in zip(bars3, region_qa.values):
    ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
             f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

# 1d. Sales by Region
ax4 = fig.add_subplot(gs[1, 1])
region_sales = df.groupby("region")["actual_sales"].sum().sort_values(ascending=False)
ax4.bar(region_sales.index, region_sales.values / 1e6, color=PALETTE, edgecolor="white")
ax4.set_title("Total Sales by Region", fontweight="bold")
ax4.set_ylabel("Sales ($ Millions)")

# 1e. Calls vs Conversion scatter
ax5 = fig.add_subplot(gs[1, 2])
rep_agg = df.groupby("rep_id").agg(
    avg_calls=("calls_made", "mean"),
    avg_conversion=("conversion_rate_pct", "mean"),
    avg_attainment=("quota_attainment_pct", "mean")
).reset_index()
sc = ax5.scatter(rep_agg["avg_calls"], rep_agg["avg_conversion"],
                 c=rep_agg["avg_attainment"], cmap="RdYlGn",
                 s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
plt.colorbar(sc, ax=ax5, label="Quota Attainment %")
ax5.set_title("Calls vs Conversion Rate (per Rep)", fontweight="bold")
ax5.set_xlabel("Avg Monthly Calls")
ax5.set_ylabel("Avg Conversion Rate %")

plt.savefig(OUT + "fig1_executive_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 1 saved")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Rep Performance Segmentation
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Sales Rep Performance Segmentation", fontsize=16, fontweight="bold")

# 2a. Top 10 vs Bottom 10 reps
rep_total = df.groupby("rep_id")["actual_sales"].sum().sort_values(ascending=False)
top10 = rep_total.head(10)
bot10 = rep_total.tail(10)

ax = axes[0]
y = np.arange(10)
ax.barh(y, top10.values / 1e6, color=PALETTE[0], label="Top 10", alpha=0.9)
ax.barh(y - 0.4, bot10.values / 1e6, color=PALETTE[3], label="Bottom 10", alpha=0.7, height=0.35)
ax.set_yticks(y)
ax.set_yticklabels(top10.index, fontsize=8)
ax.set_title("Top 10 vs Bottom 10 Reps\n(Total Sales)", fontweight="bold")
ax.set_xlabel("Total Sales ($ Millions)")
ax.legend()

# 2b. Experience vs performance
ax = axes[1]
rep_perf = df.groupby("rep_id").agg(
    experience=("rep_experience_years", "first"),
    avg_attainment=("quota_attainment_pct", "mean")
).reset_index()
exp_bins = pd.cut(rep_perf["experience"], bins=[0, 3, 7, 11, 16],
                  labels=["1-3 yrs", "4-7 yrs", "8-11 yrs", "12+ yrs"])
exp_group = rep_perf.groupby(exp_bins, observed=True)["avg_attainment"].mean()
ax.bar(exp_group.index.astype(str), exp_group.values,
       color=PALETTE[:4], edgecolor="white")
ax.axhline(100, color="black", linestyle="--", linewidth=1.5)
ax.set_title("Avg Quota Attainment\nby Experience Band", fontweight="bold")
ax.set_ylabel("Quota Attainment %")
ax.set_xlabel("Experience Band")

# 2c. Heatmap: Drug × Region performance
ax = axes[2]
heatmap_data = df.groupby(["drug", "region"])["quota_attainment_pct"].mean().unstack()
sns.heatmap(heatmap_data, ax=ax, cmap="RdYlGn", center=100,
            annot=True, fmt=".1f", linewidths=0.5,
            cbar_kws={"label": "Quota Attainment %"})
ax.set_title("Quota Attainment %\nDrug × Region Heatmap", fontweight="bold")
ax.set_xlabel("Region")
ax.set_ylabel("Drug")

plt.tight_layout()
plt.savefig(OUT + "fig2_rep_segmentation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 2 saved")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — XGBoost Sales Forecasting
# ═══════════════════════════════════════════════════════════════════════

# Feature engineering
df_model = df.copy()
le_drug = LabelEncoder()
le_region = LabelEncoder()
df_model["drug_enc"] = le_drug.fit_transform(df_model["drug"])
df_model["region_enc"] = le_region.fit_transform(df_model["region"])
df_model["quarter"] = ((df_model["month"] - 1) // 3) + 1
df_model["month_sin"] = np.sin(2 * np.pi * df_model["month"] / 12)
df_model["month_cos"] = np.cos(2 * np.pi * df_model["month"] / 12)

FEATURES = ["drug_enc", "region_enc", "year", "month", "quarter",
            "month_sin", "month_cos", "calls_made",
            "conversion_rate_pct", "rep_experience_years"]
TARGET = "actual_sales"

X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   MAE  = ${mae:,.0f}")
print(f"   RMSE = ${rmse:,.0f}")
print(f"   R²   = {r2:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"XGBoost Sales Forecasting Model  |  R² = {r2:.3f} | RMSE = ${rmse:,.0f}",
             fontsize=15, fontweight="bold")

# 3a. Actual vs Predicted scatter
ax = axes[0]
lim_min = min(y_test.min(), y_pred.min()) * 0.95
lim_max = max(y_test.max(), y_pred.max()) * 1.05
ax.scatter(y_test, y_pred, alpha=0.3, s=15, color=PALETTE[0])
ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=2, label="Perfect Fit")
ax.set_title("Actual vs Predicted Sales", fontweight="bold")
ax.set_xlabel("Actual Sales ($)")
ax.set_ylabel("Predicted Sales ($)")
ax.legend()
ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
        fontsize=12, fontweight="bold", color=PALETTE[2])

# 3b. Residuals
ax = axes[1]
residuals = y_test.values - y_pred
ax.hist(residuals, bins=50, color=PALETTE[1], edgecolor="white", alpha=0.85)
ax.axvline(0, color="black", linestyle="--", linewidth=2)
ax.set_title("Residuals Distribution", fontweight="bold")
ax.set_xlabel("Residual ($)")
ax.set_ylabel("Count")
ax.text(0.65, 0.9, f"Mean: ${residuals.mean():,.0f}\nStd: ${residuals.std():,.0f}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# 3c. Feature Importance
ax = axes[2]
feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
ax.barh(feat_imp.index, feat_imp.values, color=PALETTE[0], alpha=0.85)
ax.set_title("Feature Importance (XGBoost)", fontweight="bold")
ax.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig(OUT + "fig3_forecasting_model.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 3 saved")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Business Recommendations
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Business Recommendations & Opportunity Analysis", fontsize=16, fontweight="bold")

# 4a. Underperforming reps: gap to quota
ax = axes[0]
rep_qa = df.groupby(["rep_id", "region"])["quota_attainment_pct"].mean().reset_index()
underperformers = rep_qa[rep_qa["quota_attainment_pct"] < 95].sort_values("quota_attainment_pct")
colors_u = [PALETTE[3] if v < 85 else PALETTE[2]
            for v in underperformers["quota_attainment_pct"].values]
ax.barh(underperformers["rep_id"], underperformers["quota_attainment_pct"],
        color=colors_u, edgecolor="white")
ax.axvline(95, color="black", linestyle="--", linewidth=1.5, label="95% threshold")
ax.axvline(100, color="gray", linestyle=":", linewidth=1.5, label="Quota")
ax.set_title(f"Underperforming Reps (Attainment < 95%)\nn = {len(underperformers)}",
             fontweight="bold")
ax.set_xlabel("Avg Quota Attainment %")
ax.legend(fontsize=9)

# 4b. Seasonal opportunity — month-wise gap
ax = axes[1]
monthly_drug = df.groupby(["month", "drug"])["quota_attainment_pct"].mean().unstack()
monthly_drug.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white", width=0.75)
ax.axhline(100, color="black", linestyle="--", linewidth=1.5, label="Quota Line")
ax.set_title("Monthly Quota Attainment by Drug\n(Seasonal Pattern)", fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Quota Attainment %")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Drug", fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(OUT + "fig4_business_recommendations.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 4 saved")

# ═══════════════════════════════════════════════════════════════════════
# Save summary stats CSV for README
# ═══════════════════════════════════════════════════════════════════════
summary = df.groupby(["region", "drug"]).agg(
    total_sales=("actual_sales", "sum"),
    avg_quota_attainment=("quota_attainment_pct", "mean"),
    avg_calls=("calls_made", "mean"),
    avg_conversion=("conversion_rate_pct", "mean"),
    num_reps=("rep_id", "nunique")
).round(2).reset_index()
summary.to_csv(OUT + "summary_by_region_drug.csv", index=False)

print(f"\n✅ All outputs saved to {OUT}")
print(f"\n📈 Key Business Insights:")
print(f"   • East region leads quota attainment at {df[df['region']=='East']['quota_attainment_pct'].mean():.1f}%")
print(f"   • Central region lags at {df[df['region']=='Central']['quota_attainment_pct'].mean():.1f}%")
print(f"   • {len(underperformers)} reps below 95% quota attainment — intervention needed")
print(f"   • OncoPrime drives highest revenue at ${df[df['drug']=='OncoPrime']['actual_sales'].sum()/1e6:.1f}M total")
print(f"   • Model R² = {r2:.3f} — {r2*100:.1f}% variance in sales explained by features")
