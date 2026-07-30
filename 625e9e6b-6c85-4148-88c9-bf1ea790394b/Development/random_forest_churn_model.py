
# ─── Configuration ───────────────────────────────────────────────────────────
N_ESTIMATORS       = 300
RANDOM_STATE       = 42
CLASS_WEIGHT       = "balanced"
FEATURE_NAMES      = ["Frequency", "Monetary"]
TARGET_NAMES       = ["Active", "Churned"]

# LR baseline metrics (from churn_prediction_model block)
LR_ACCURACY  = 0.6956
LR_PRECISION = 0.6297
LR_RECALL    = 0.8671
LR_F1        = 0.7278

# LR standardised coefficients (absolute values for comparison)
LR_COEF_FREQ = abs(-0.732)
LR_COEF_MON  = abs(-0.327)

# ─── Imports ─────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cycler import cycler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# ─── 1. Train Random Forest ───────────────────────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    class_weight=CLASS_WEIGHT,
    n_jobs=-1
)
rf_model.fit(X_train_sc, y_train)

# ─── 2. Predict ───────────────────────────────────────────────────────────────
y_pred_rf      = rf_model.predict(X_test_sc)
y_pred_prob_rf = rf_model.predict_proba(X_test_sc)[:, 1]

# ─── 3. Compute metrics ───────────────────────────────────────────────────────
rf_accuracy  = accuracy_score(y_test,  y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, zero_division=0)
rf_recall    = recall_score(y_test,    y_pred_rf, zero_division=0)
rf_f1        = f1_score(y_test,        y_pred_rf, zero_division=0)

# ─── 4. Feature importances ───────────────────────────────────────────────────
importances   = rf_model.feature_importances_
rf_importances = {feat: imp for feat, imp in zip(FEATURE_NAMES, importances)}

# ─── 5. Console report ────────────────────────────────────────────────────────
SEP = "=" * 64

print(SEP)
print("  RANDOM FOREST — CHURN PREDICTION PERFORMANCE")
print(SEP)
print(f"  Accuracy  : {rf_accuracy:.4f}")
print(f"  Precision : {rf_precision:.4f}")
print(f"  Recall    : {rf_recall:.4f}")
print(f"  F1-score  : {rf_f1:.4f}")
print()
print("─── Classification Report ──────────────────────────────────────")
print(classification_report(y_test, y_pred_rf, target_names=TARGET_NAMES))

print(SEP)
print("  MODEL COMPARISON")
print(SEP)
header = f"  {'Metric':<12} {'Log. Regression':>18} {'Random Forest':>16}  {'Δ (RF−LR)':>12}"
print(header)
print("  " + "─" * (len(header) - 2))
metrics = [
    ("Accuracy",  LR_ACCURACY,  rf_accuracy),
    ("Precision", LR_PRECISION, rf_precision),
    ("Recall",    LR_RECALL,    rf_recall),
    ("F1-score",  LR_F1,        rf_f1),
]
for label, lr_val, rf_val in metrics:
    delta = rf_val - lr_val
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<12} {lr_val:>18.4f} {rf_val:>16.4f}  {sign}{delta:>11.4f}")
print(SEP)
print()
print("  RF Feature Importances:")
for feat, imp in rf_importances.items():
    print(f"    {feat:<12}: {imp:.4f}")
print(SEP)

# ─── 6. Figure: 2×2 grid ──────────────────────────────────────────────────────
PALETTE = ["#3D5A80", "#B07D62", "#6B8F71", "#8C6A9E"]

fig_rf_comparison, axes_grid = plt.subplots(
    2, 2, figsize=(14, 11), constrained_layout=True
)
fig_rf_comparison.set_dpi(200)
fig_rf_comparison.suptitle(
    "Random Forest Churn Model — Performance & Comparison",
    fontweight="bold", fontsize=14
)

# ── Top-left: Confusion Matrix ────────────────────────────────────────────────
ax_cm = axes_grid[0, 0]
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_NAMES)
disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
ax_cm.set_title("RF Confusion Matrix", fontsize=11, fontweight="bold")
ax_cm.spines[["top", "right"]].set_visible(False)

# ── Top-right: Grouped bar chart LR vs RF metrics ────────────────────────────
ax_bar = axes_grid[0, 1]
metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
lr_vals = [LR_ACCURACY, LR_PRECISION, LR_RECALL, LR_F1]
rf_vals = [rf_accuracy, rf_precision, rf_recall, rf_f1]

x        = np.arange(len(metric_labels))
bar_w    = 0.35
b1 = ax_bar.bar(x - bar_w / 2, lr_vals, bar_w, label="Logistic Regression",
                color=PALETTE[0], alpha=0.85)
b2 = ax_bar.bar(x + bar_w / 2, rf_vals, bar_w, label="Random Forest",
                color=PALETTE[1], alpha=0.85)

# Value labels on bars
for rect, val in zip(b1, lr_vals):
    ax_bar.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color=PALETTE[0])
for rect, val in zip(b2, rf_vals):
    ax_bar.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color=PALETTE[1])

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metric_labels, fontsize=10)
ax_bar.set_ylabel("Score", fontsize=10)
ax_bar.set_ylim(0, 1.12)
ax_bar.set_title("Model Comparison: Logistic Regression vs Random Forest",
                 fontsize=10, fontweight="bold")
ax_bar.legend(frameon=False, fontsize=9)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
ax_bar.set_axisbelow(True)

# ── Bottom-left: RF Feature Importances (horizontal bar) ─────────────────────
ax_fi = axes_grid[1, 0]
y_pos = np.arange(len(FEATURE_NAMES))
bars_fi = ax_fi.barh(y_pos, importances, color=PALETTE[2], alpha=0.85)
ax_fi.set_yticks(y_pos)
ax_fi.set_yticklabels(FEATURE_NAMES, fontsize=10)
ax_fi.set_xlabel("Importance", fontsize=10)
ax_fi.set_title("Random Forest — Feature Importances", fontsize=10, fontweight="bold")
for rect, val in zip(bars_fi, importances):
    ax_fi.text(val + 0.002, rect.get_y() + rect.get_height() / 2,
               f"{val:.4f}", va="center", fontsize=9)
ax_fi.spines[["top", "right"]].set_visible(False)
ax_fi.xaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
ax_fi.set_axisbelow(True)

# ── Bottom-right: LR |coefs| vs RF Importances (grouped horizontal bars) ──────
ax_cmp = axes_grid[1, 1]
lr_abs_coefs = [LR_COEF_FREQ, LR_COEF_MON]   # |Frequency|, |Monetary|
rf_imps      = [rf_importances[f] for f in FEATURE_NAMES]

y_pos2 = np.arange(len(FEATURE_NAMES))
bh_w   = 0.35
b_lr = ax_cmp.barh(y_pos2 + bh_w / 2, lr_abs_coefs, bh_w,
                   label="LR |Coefficient|", color=PALETTE[0], alpha=0.85)
b_rf = ax_cmp.barh(y_pos2 - bh_w / 2, rf_imps, bh_w,
                   label="RF Importance",   color=PALETTE[1], alpha=0.85)

for rect, val in zip(b_lr, lr_abs_coefs):
    ax_cmp.text(val + 0.005, rect.get_y() + rect.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5, color=PALETTE[0])
for rect, val in zip(b_rf, rf_imps):
    ax_cmp.text(val + 0.005, rect.get_y() + rect.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5, color=PALETTE[1])

ax_cmp.set_yticks(y_pos2)
ax_cmp.set_yticklabels(FEATURE_NAMES, fontsize=10)
ax_cmp.set_xlabel("Weight / Importance", fontsize=10)
ax_cmp.set_title("Feature Weights: LR Coefficients (|abs|) vs RF Importances",
                 fontsize=10, fontweight="bold")
ax_cmp.legend(frameon=False, fontsize=9)
ax_cmp.spines[["top", "right"]].set_visible(False)
ax_cmp.xaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
ax_cmp.set_axisbelow(True)

# ─── Finalise ─────────────────────────────────────────────────────────────────
fig_rf_comparison = fig_rf_comparison   # assign for export
plt.close("all")                        # suppress auto-display; named var above keeps it

print("\nFigure 'fig_rf_comparison' ready.")
print(f"\nExported variables: rf_accuracy={rf_accuracy:.4f}, rf_precision={rf_precision:.4f}, "
      f"rf_recall={rf_recall:.4f}, rf_f1={rf_f1:.4f}")
print(f"rf_importances: {rf_importances}")
print("\nNOTE: rfm_clv['Churn_Probability'] left intact (LR probabilities unchanged).")
