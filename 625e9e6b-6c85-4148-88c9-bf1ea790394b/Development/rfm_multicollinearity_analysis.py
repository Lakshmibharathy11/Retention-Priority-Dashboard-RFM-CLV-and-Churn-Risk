import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── 1. Extract Frequency & Monetary from the RFM table ───────────────────────
freq = rfm["Frequency"].values
mon  = rfm["Monetary"].values

# ── 2. Pearson Correlation ────────────────────────────────────────────────────
pearson_r, pearson_p = stats.pearsonr(freq, mon)

print("=" * 55)
print("  Pearson Correlation — Frequency vs Monetary")
print("=" * 55)
print(f"  r  = {pearson_r:+.4f}")
print(f"  p  = {pearson_p:.2e}")
strength = (
    "very strong" if abs(pearson_r) >= 0.8 else
    "strong"      if abs(pearson_r) >= 0.6 else
    "moderate"    if abs(pearson_r) >= 0.4 else
    "weak"        if abs(pearson_r) >= 0.2 else
    "negligible"
)
sig = "statistically significant (p < 0.05)" if pearson_p < 0.05 else "not statistically significant"
print(f"\n  → {strength.capitalize()} positive correlation, {sig}.")

# ── 3. VIF ────────────────────────────────────────────────────────────────────
# For exactly two features, VIF = 1 / (1 - r²) — mathematically exact
r_sq = pearson_r ** 2
vif_value = 1 / (1 - r_sq)   # identical for both features when there are only 2
vif_freq = round(vif_value, 4)
vif_mon  = round(vif_value, 4)

vif_df = pd.DataFrame({
    "Feature": ["Frequency", "Monetary"],
    "VIF":     [round(vif_freq, 4), round(vif_mon, 4)]
})

print("\n" + "=" * 55)
print("  Variance Inflation Factor (VIF)")
print("=" * 55)
print(vif_df.to_string(index=False))
print()
for _, row_vif in vif_df.iterrows():
    level = (
        "severe multicollinearity (>10)"   if row_vif["VIF"] > 10 else
        "moderate multicollinearity (5–10)" if row_vif["VIF"] > 5 else
        "acceptable (≤ 5)"
    )
    print(f"  {row_vif['Feature']:>10s}: VIF = {row_vif['VIF']:.4f}  → {level}")

# ── 4. Visualisation ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Frequency vs Monetary — Multicollinearity Assessment", fontsize=14, fontweight="bold")

# Scatter with regression line
ax_s = axes[0]
ax_s.scatter(freq, mon, alpha=0.25, s=12, color="#4C72B0", edgecolors="none")
m, b = np.polyfit(freq, mon, 1)
x_line = np.linspace(freq.min(), freq.max(), 200)
ax_s.plot(x_line, m * x_line + b, color="#DD4444", lw=2, label=f"r = {pearson_r:+.4f}")
ax_s.set_xlabel("Frequency (# orders)", fontsize=11)
ax_s.set_ylabel("Monetary (£)", fontsize=11)
ax_s.set_title("Scatter Plot with Regression Line", fontsize=12)
ax_s.legend(fontsize=11)
ax_s.grid(True, alpha=0.3)

# VIF bar chart
ax_v = axes[1]
colours = ["#2ecc71" if v <= 5 else "#e67e22" if v <= 10 else "#e74c3c"
           for v in vif_df["VIF"]]
bars_vif = ax_v.barh(vif_df["Feature"], vif_df["VIF"], color=colours, edgecolor="white")
ax_v.axvline(5,  color="#e67e22", lw=1.5, ls="--", label="VIF = 5  (moderate threshold)")
ax_v.axvline(10, color="#e74c3c", lw=1.5, ls="--", label="VIF = 10 (severe threshold)")
for bar_v, val_v in zip(bars_vif, vif_df["VIF"]):
    ax_v.text(bar_v.get_width() + 0.05, bar_v.get_y() + bar_v.get_height() / 2,
              f"{val_v:.2f}", va="center", fontsize=11, fontweight="bold")
ax_v.set_xlabel("VIF", fontsize=11)
ax_v.set_title("Variance Inflation Factor per Feature", fontsize=12)
ax_v.legend(fontsize=9)
ax_v.grid(True, axis="x", alpha=0.3)
ax_v.set_xlim(0, max(vif_df["VIF"]) * 1.3)

plt.tight_layout()
fig_multicollinearity = fig
plt.close("all")