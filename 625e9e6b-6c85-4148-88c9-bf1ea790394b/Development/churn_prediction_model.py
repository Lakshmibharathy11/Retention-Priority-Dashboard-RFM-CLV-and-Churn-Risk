import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# ── 1. Build features & target ────────────────────────────────────────────────
# Recency is EXCLUDED — it was used to define the churn label (data leakage fix)
CHURN_DAYS = CHURN_THRESHOLD_DAYS  # 138 days, defined upstream

features = ["Frequency", "Monetary"]          # Recency intentionally dropped
X = rfm[features].copy()
y = (rfm["Recency"] > CHURN_DAYS).astype(int)  # 1 = churned

churn_rate_pct = y.mean() * 100
print(f"Churn label distribution: {y.value_counts().to_dict()}")
print(f"Churn rate in dataset   : {churn_rate_pct:.1f}%")
print(f"\nFeatures used: {features}  (Recency excluded — leakage prevention)")

# ── 2. Train / test split (80/20, stratified) ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ── 3. Scale features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Train logistic regression ──────────────────────────────────────────────
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr_model.fit(X_train_sc, y_train)

# ── 5. Evaluate on test set ───────────────────────────────────────────────────
y_pred      = lr_model.predict(X_test_sc)
y_pred_prob = lr_model.predict_proba(X_test_sc)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)

print("\n── Model Performance (Test Set) — Frequency + Monetary only ──────────")
print(f"  Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  Precision: {precision:.4f}  ({precision*100:.1f}%)")
print(f"  Recall   : {recall:.4f}  ({recall*100:.1f}%)")
print(f"\nFull classification report:\n{classification_report(y_test, y_pred, target_names=['Active','Churned'])}")

# ── 6. Confusion matrix + coefficient plot ────────────────────────────────────
fig_churn_model, axes = plt.subplots(1, 2, figsize=(13, 5))
fig_churn_model.suptitle(
    "Logistic Regression — Churn Prediction\n(Features: Frequency & Monetary  |  Recency excluded — data leakage fix)",
    fontsize=13, fontweight="bold", y=1.03
)

# Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Active (0)", "Churned (1)"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix", fontweight="bold")

# Feature coefficients
coefs       = lr_model.coef_[0]
feat_labels = features
colors      = ["#e74c3c" if c > 0 else "#2ecc71" for c in coefs]
bars_h      = axes[1].barh(feat_labels, coefs, color=colors, edgecolor="white", height=0.4)
axes[1].axvline(0, color="grey", linewidth=0.8, linestyle="--")
axes[1].set_xlabel("Coefficient (standardised)")
axes[1].set_title("Feature Coefficients\n(red = increases churn risk)", fontweight="bold")
for bar, val in zip(bars_h, coefs):
    axes[1].text(
        val + (0.02 if val >= 0 else -0.02),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.3f}", va="center",
        ha="left" if val >= 0 else "right", fontsize=11
    )

plt.tight_layout()
plt.close("all")

# ── 7. Score full customer base & update rfm_clv ─────────────────────────────
X_full_sc       = scaler.transform(rfm[features])
churn_proba_all = lr_model.predict_proba(X_full_sc)[:, 1]

# Overwrite column so rfm_clv stays consistent with the leak-free model
rfm_clv["Churn_Probability"] = churn_proba_all

print("\n── Churn_Probability (leak-free) added to rfm_clv ──────────────────")
print(rfm_clv[["CustomerID", "Recency", "Frequency", "Monetary", "Segment", "Churn_Probability"]].head(10).to_string(index=False))

# ── 8. Churn probability distribution ────────────────────────────────────────
fig_churn_proba, ax_p = plt.subplots(figsize=(9, 4))
ax_p.hist(rfm_clv["Churn_Probability"], bins=40, color="#3498db", edgecolor="white", alpha=0.85)
ax_p.axvline(0.5, color="#e74c3c", linewidth=1.5, linestyle="--", label="Decision threshold (0.5)")
ax_p.set_xlabel("Predicted Churn Probability")
ax_p.set_ylabel("Number of Customers")
ax_p.set_title(
    "Distribution of Predicted Churn Probabilities (full customer base)\n"
    "Model: Frequency + Monetary  |  Recency excluded",
    fontweight="bold"
)
ax_p.legend()
plt.tight_layout()
plt.close("all")

print(f"\nPredicted high-risk customers (prob > 0.5): {(rfm_clv['Churn_Probability'] > 0.5).sum():,}")
print(f"Predicted low-risk  customers (prob ≤ 0.5): {(rfm_clv['Churn_Probability'] <= 0.5).sum():,}")