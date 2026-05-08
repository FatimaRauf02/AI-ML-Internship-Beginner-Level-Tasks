"""
Task 6: House Price Prediction
Uses a synthetic dataset modeled after Kaggle House Price data.
Models: Linear Regression + Gradient Boosting
Evaluation: MAE, RMSE
Visualizations: Actual vs Predicted, Feature Importance, Residuals, Price Distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

# ─── 1. Generate Realistic Dataset ────────────────────────────────────────────
np.random.seed(42)
n = 1000

locations = ["Downtown", "Suburbs", "Rural", "Waterfront", "Midtown"]
loc_base  = {"Downtown": 180000, "Suburbs": 140000, "Rural": 90000,
              "Waterfront": 250000, "Midtown": 160000}

data = {
    "sqft"      : np.random.randint(500, 5000, n),
    "bedrooms"  : np.random.randint(1, 7, n),
    "bathrooms" : np.random.randint(1, 5, n),
    "age"       : np.random.randint(0, 80, n),
    "garage"    : np.random.randint(0, 4, n),
    "location"  : np.random.choice(locations, n),
    "floors"    : np.random.randint(1, 4, n),
    "pool"      : np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)
df["price"] = (
    df["location"].map(loc_base)
    + df["sqft"]      * 85
    + df["bedrooms"]  * 8000
    + df["bathrooms"] * 6000
    - df["age"]       * 500
    + df["garage"]    * 7000
    + df["floors"]    * 5000
    + df["pool"]      * 15000
    + np.random.normal(0, 18000, n)
).clip(50000)

print("=" * 60)
print("  🏠  House Price Prediction — Task 6")
print("=" * 60)
print(f"\n📊 Dataset: {len(df)} records, {df.shape[1]} features")
print("\nSample Data:")
print(df.head())
print("\nBasic Stats:")
print(df[["sqft","bedrooms","bathrooms","age","price"]].describe().round(0))

# ─── 2. Preprocessing ─────────────────────────────────────────────────────────
print("\n⚙️  Preprocessing...")

le = LabelEncoder()
df["location_enc"] = le.fit_transform(df["location"])

FEATURES = ["sqft","bedrooms","bathrooms","age","garage","location_enc","floors","pool"]
TARGET   = "price"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ─── 3. Train Models ──────────────────────────────────────────────────────────
print("\n🤖 Training models...")

lr  = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                max_depth=4, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

# ─── 4. Evaluate ──────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2))
    print(f"\n  📈 {name}")
    print(f"     MAE  : ${mae:,.0f}")
    print(f"     RMSE : ${rmse:,.0f}")
    print(f"     R²   : {r2:.4f}")
    return mae, rmse, r2

print("\n📉 Evaluation Results:")
mae_lr, rmse_lr, r2_lr = evaluate("Linear Regression", y_test, y_pred_lr)
mae_gb, rmse_gb, r2_gb = evaluate("Gradient Boosting", y_test, y_pred_gb)

# ─── 5. Visualizations ────────────────────────────────────────────────────────
print("\n🎨 Generating visualizations...")

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"lr": "#4A90D9", "gb": "#E8543A", "actual": "#2ECC71", "neutral": "#7F8C8D"}

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#F8F9FA")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

fmt = FuncFormatter(lambda x, _: f"${x/1000:.0f}K")

# ── Plot 1: Actual vs Predicted — Linear Regression ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_pred_lr, alpha=0.4, color=COLORS["lr"], s=18, label="Predictions")
mn, mx = y_test.min(), y_test.max()
ax1.plot([mn,mx],[mn,mx], "r--", lw=1.5, label="Perfect Fit")
ax1.set_title("Linear Regression\nActual vs Predicted", fontsize=11, fontweight="bold")
ax1.set_xlabel("Actual Price"); ax1.set_ylabel("Predicted Price")
ax1.xaxis.set_major_formatter(fmt); ax1.yaxis.set_major_formatter(fmt)
ax1.legend(fontsize=8)
ax1.text(0.05, 0.92, f"R²={r2_lr:.3f}", transform=ax1.transAxes,
         fontsize=9, color=COLORS["lr"], fontweight="bold")

# ── Plot 2: Actual vs Predicted — Gradient Boosting ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, y_pred_gb, alpha=0.4, color=COLORS["gb"], s=18)
ax2.plot([mn,mx],[mn,mx], "r--", lw=1.5)
ax2.set_title("Gradient Boosting\nActual vs Predicted", fontsize=11, fontweight="bold")
ax2.set_xlabel("Actual Price"); ax2.set_ylabel("Predicted Price")
ax2.xaxis.set_major_formatter(fmt); ax2.yaxis.set_major_formatter(fmt)
ax2.text(0.05, 0.92, f"R²={r2_gb:.3f}", transform=ax2.transAxes,
         fontsize=9, color=COLORS["gb"], fontweight="bold")

# ── Plot 3: MAE & RMSE Comparison ──
ax3 = fig.add_subplot(gs[0, 2])
models  = ["Linear\nRegression", "Gradient\nBoosting"]
maes    = [mae_lr/1000, mae_gb/1000]
rmses   = [rmse_lr/1000, rmse_gb/1000]
x       = np.arange(len(models))
w       = 0.35
b1 = ax3.bar(x - w/2, maes,  w, label="MAE",  color=COLORS["lr"], alpha=0.85)
b2 = ax3.bar(x + w/2, rmses, w, label="RMSE", color=COLORS["gb"], alpha=0.85)
ax3.set_title("MAE vs RMSE\nModel Comparison", fontsize=11, fontweight="bold")
ax3.set_ylabel("Error ($K)"); ax3.set_xticks(x); ax3.set_xticklabels(models)
ax3.legend(fontsize=9)
for bar in list(b1)+list(b2):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"${bar.get_height():.1f}K", ha="center", va="bottom", fontsize=8)

# ── Plot 4: Residuals — Linear Regression ──
ax4 = fig.add_subplot(gs[1, 0])
res_lr = y_test.values - y_pred_lr
ax4.scatter(y_pred_lr, res_lr, alpha=0.35, color=COLORS["lr"], s=15)
ax4.axhline(0, color="red", linestyle="--", lw=1.5)
ax4.set_title("Linear Regression\nResiduals", fontsize=11, fontweight="bold")
ax4.set_xlabel("Predicted Price"); ax4.set_ylabel("Residual")
ax4.xaxis.set_major_formatter(fmt)
ax4.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))

# ── Plot 5: Residuals — Gradient Boosting ──
ax5 = fig.add_subplot(gs[1, 1])
res_gb = y_test.values - y_pred_gb
ax5.scatter(y_pred_gb, res_gb, alpha=0.35, color=COLORS["gb"], s=15)
ax5.axhline(0, color="red", linestyle="--", lw=1.5)
ax5.set_title("Gradient Boosting\nResiduals", fontsize=11, fontweight="bold")
ax5.set_xlabel("Predicted Price"); ax5.set_ylabel("Residual")
ax5.xaxis.set_major_formatter(fmt)
ax5.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))

# ── Plot 6: Feature Importance (GB) ──
ax6 = fig.add_subplot(gs[1, 2])
feat_names  = ["Sq.Ft","Bedrooms","Bathrooms","Age","Garage","Location","Floors","Pool"]
importances = gb.feature_importances_
idx = np.argsort(importances)
colors_bar  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(idx)))
ax6.barh([feat_names[i] for i in idx], importances[idx], color=colors_bar, edgecolor="white")
ax6.set_title("Feature Importance\n(Gradient Boosting)", fontsize=11, fontweight="bold")
ax6.set_xlabel("Importance Score")
for i, (v, ix) in enumerate(zip(importances[idx], idx)):
    ax6.text(v+0.001, i, f"{v:.3f}", va="center", fontsize=8)

# ── Plot 7: Price Distribution by Location ──
ax7 = fig.add_subplot(gs[2, 0])
for loc in locations:
    prices = df[df["location"]==loc]["price"]
    ax7.hist(prices/1000, bins=25, alpha=0.55, label=loc, density=True)
ax7.set_title("Price Distribution\nby Location", fontsize=11, fontweight="bold")
ax7.set_xlabel("Price ($K)"); ax7.set_ylabel("Density")
ax7.legend(fontsize=7)

# ── Plot 8: Price vs Sqft ──
ax8 = fig.add_subplot(gs[2, 1])
sc = ax8.scatter(df["sqft"], df["price"]/1000, c=df["bedrooms"],
                 cmap="viridis", alpha=0.35, s=12)
plt.colorbar(sc, ax=ax8, label="Bedrooms")
ax8.set_title("Price vs Square Footage\n(colored by Bedrooms)", fontsize=11, fontweight="bold")
ax8.set_xlabel("Square Footage"); ax8.set_ylabel("Price ($K)")

# ── Plot 9: R² Score Comparison ──
ax9 = fig.add_subplot(gs[2, 2])
r2s    = [r2_lr, r2_gb]
colors = [COLORS["lr"], COLORS["gb"]]
bars   = ax9.bar(["Linear\nRegression","Gradient\nBoosting"], r2s, color=colors, alpha=0.85, width=0.4)
ax9.set_title("R² Score\nComparison", fontsize=11, fontweight="bold")
ax9.set_ylabel("R² Score"); ax9.set_ylim(0, 1.05)
for bar, val in zip(bars, r2s):
    ax9.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f"{val:.4f}", ha="center", fontweight="bold", fontsize=10)

fig.suptitle("🏠 House Price Prediction — Model Analysis Dashboard",
             fontsize=16, fontweight="bold", y=0.98)

output_path = "house_price_results.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"   ✅ Saved: {output_path}")

# ─── 6. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  📋  FINAL SUMMARY")
print("=" * 60)
print(f"\n  {'Metric':<12} {'Linear Reg':>15} {'Gradient Boost':>15}")
print(f"  {'-'*42}")
print(f"  {'MAE':<12} ${mae_lr:>13,.0f} ${mae_gb:>13,.0f}")
print(f"  {'RMSE':<12} ${rmse_lr:>13,.0f} ${rmse_gb:>13,.0f}")
print(f"  {'R²':<12} {r2_lr:>15.4f} {r2_gb:>15.4f}")
print(f"\n  🏆 Best Model: {'Gradient Boosting' if r2_gb > r2_lr else 'Linear Regression'}")
print(f"  📁 Chart saved to: {output_path}")
print("=" * 60)