
# ── Step 2 — Exploratory Data Analysis ────────────────────────────────────
# Working on a cleaned subset: no cancellations, qty>0, price>0, customer known.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cycler import cycler
from matplotlib.patches import FancyBboxPatch

# ── Palette / style ──────────────────────────────────────────────────────────
PALETTE = ['#3D5A80', '#B07D62', '#6B8F71', '#8C6A9E', '#C9A227', '#4C8C8C']
plt.rcParams.update({
    'axes.prop_cycle'  : cycler(color=PALETTE),
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
})

# ── Clean subset ─────────────────────────────────────────────────────────────
df_clean = (
    df_raw
    .loc[
        (~df_raw[INVOICE_COL].astype(str).str.startswith('C'))
        & (df_raw[QTY_COL]   >  0)
        & (df_raw[PRICE_COL] >  0)
        & (df_raw[CUSTOMER_COL].notna())
    ]
    .copy()
)
df_clean[REVENUE_COL] = df_clean[QTY_COL] * df_clean[PRICE_COL]

print("=" * 65)
print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 65)
print(f"\n  Rows in raw dataset    : {len(df_raw):>10,}")
print(f"  Rows removed (filtered): {len(df_raw) - len(df_clean):>10,}")
print(f"  Rows in clean subset   : {len(df_clean):>10,}  ({len(df_clean)/len(df_raw)*100:.1f}% of raw)")

# ─────────────────────────────────────────────────────────────────────────────
# A. DATE RANGE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  A. DATE RANGE")
print(f"{'─'*65}")
date_min = df_clean[DATE_COL].min()
date_max = df_clean[DATE_COL].max()
print(f"  Earliest transaction : {date_min}")
print(f"  Latest  transaction  : {date_max}")
print(f"  Span                 : {(date_max - date_min).days} days  (~{(date_max - date_min).days/365.25:.1f} years)")

# ─────────────────────────────────────────────────────────────────────────────
# B. UNIQUE COUNTS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  B. UNIQUE COUNTS")
print(f"{'─'*65}")
n_customers = df_clean[CUSTOMER_COL].nunique()
n_products  = df_clean[STOCK_COL].nunique()
n_invoices  = df_clean[INVOICE_COL].nunique()
print(f"  Unique customers : {n_customers:>8,}")
print(f"  Unique products  : {n_products:>8,}")
print(f"  Unique invoices  : {n_invoices:>8,}")

# ─────────────────────────────────────────────────────────────────────────────
# C. ORDER VALUE (sum of LineRevenue per Invoice)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  C. ORDER VALUE  (Quantity × Price summed per Invoice)")
print(f"{'─'*65}")
order_value = df_clean.groupby(INVOICE_COL)[REVENUE_COL].sum()
ov_stats = order_value.describe(percentiles=[.25, .5, .75])
print(f"  Mean   : £{ov_stats['mean']:>10.2f}")
print(f"  Median : £{ov_stats['50%']:>10.2f}")
print(f"  Std    : £{ov_stats['std']:>10.2f}")
print(f"  25th % : £{ov_stats['25%']:>10.2f}")
print(f"  75th % : £{ov_stats['75%']:>10.2f}")
print(f"  Max    : £{ov_stats['max']:>10.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# D. ORDER FREQUENCY PER CUSTOMER
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  D. ORDER FREQUENCY PER CUSTOMER  (distinct invoices)")
print(f"{'─'*65}")
order_freq = df_clean.groupby(CUSTOMER_COL)[INVOICE_COL].nunique()
of_stats = order_freq.describe(percentiles=[.25, .5, .75])
print(f"  Mean   : {of_stats['mean']:>8.2f}  orders")
print(f"  Median : {of_stats['50%']:>8.2f}  orders")
print(f"  Std    : {of_stats['std']:>8.2f}  orders")
print(f"  25th % : {of_stats['25%']:>8.2f}  orders")
print(f"  75th % : {of_stats['75%']:>8.2f}  orders")
print(f"  Max    : {of_stats['max']:>8.0f}  orders")

# ─────────────────────────────────────────────────────────────────────────────
# E. INTER-PURCHASE GAP
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  E. INTER-PURCHASE GAP  (days between consecutive orders)")
print(f"{'─'*65}")
# Per customer, get unique invoice dates, sort, diff
customer_dates = (
    df_clean[[CUSTOMER_COL, DATE_COL, INVOICE_COL]]
    .drop_duplicates(subset=[CUSTOMER_COL, INVOICE_COL])
    .groupby(CUSTOMER_COL)[DATE_COL]
    .apply(lambda s: sorted(s.tolist()))
)

gap_days_list = []
for cust, dates in customer_dates.items():
    if len(dates) > 1:
        for i in range(1, len(dates)):
            gap_days_list.append((dates[i] - dates[i-1]).days)

gap_series = pd.Series(gap_days_list, name='gap_days')
gap_p25  = gap_series.quantile(0.25)
gap_med  = gap_series.median()
gap_mean = gap_series.mean()
gap_p75  = gap_series.quantile(0.75)

print(f"  Mean gap   : {gap_mean:>8.1f} days")
print(f"  Median gap : {gap_med:>8.1f} days")
print(f"  25th %     : {gap_p25:>8.1f} days")
print(f"  75th %     : {gap_p75:>8.1f} days")
print(f"  (Pairs contributing: {len(gap_series):,})")
print(f"\n  ► Suggested churn threshold: ~{int(gap_p75 * 2)} days  (2× 75th percentile gap)")

# ─────────────────────────────────────────────────────────────────────────────
# F. MONTHLY REVENUE TREND
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  F. MONTHLY REVENUE TREND")
print(f"{'─'*65}")
df_clean['YearMonth'] = df_clean[DATE_COL].dt.to_period('M')
monthly_rev = df_clean.groupby('YearMonth')[REVENUE_COL].sum().sort_index()
print(f"  {'Month':<12} {'Revenue (£)':>14}")
print(f"  {'-'*12} {'-'*14}")
for period, rev in monthly_rev.items():
    print(f"  {str(period):<12} {rev:>14,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# G. REVENUE CONCENTRATION (Pareto)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  G. REVENUE CONCENTRATION  (Pareto / top-20% customers)")
print(f"{'─'*65}")
cust_rev = df_clean.groupby(CUSTOMER_COL)[REVENUE_COL].sum().sort_values(ascending=False)
n_top20  = max(1, int(len(cust_rev) * 0.20))
rev_top20_pct = cust_rev.iloc[:n_top20].sum() / cust_rev.sum() * 100
print(f"  Total customers             : {len(cust_rev):>8,}")
print(f"  Top 20% count               : {n_top20:>8,}")
print(f"  Revenue from top 20%        : {rev_top20_pct:>7.1f}%")
print(f"  Revenue from bottom 80%     : {100-rev_top20_pct:>7.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

# Clip order values at 99th percentile for chart 2
ov_clip_threshold = order_value.quantile(0.99)

# ── Figure 1: Order Frequency Distribution ───────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
fig1.set_dpi(200)
fig1.set_facecolor('white')
ax1.set_facecolor('white')

freq_vals = order_freq.values
ax1.hist(freq_vals, bins=80, color=PALETTE[0], alpha=0.85, edgecolor='white', linewidth=0.4)
ax1.set_yscale('log')
ax1.axvline(of_stats['50%'], color=PALETTE[1], linewidth=1.8, linestyle='--', zorder=5)
ax1.annotate(
    f"Median = {of_stats['50%']:.0f}",
    xy=(of_stats['50%'], ax1.get_ylim()[1] * 0.5),
    xytext=(of_stats['50%'] + 5, ax1.get_ylim()[1] * 0.5),
    fontsize=9, color=PALETTE[1],
    arrowprops=dict(arrowstyle='->', color=PALETTE[1], lw=1.2),
)
fig1.suptitle("Order Frequency Distribution", fontweight='bold', fontsize=13)
ax1.set_title("Orders per customer — log-scale y-axis", fontsize=10, color='dimgray')
ax1.set_xlabel("Number of Orders per Customer", fontsize=10)
ax1.set_ylabel("Number of Customers (log scale)", fontsize=10)

# ── Figure 2: Order Value Distribution ───────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5), constrained_layout=True)
fig2.set_dpi(200)
fig2.set_facecolor('white')
ax2.set_facecolor('white')

ov_clipped = order_value[order_value <= ov_clip_threshold]
ax2.hist(ov_clipped.values, bins=80, color=PALETTE[2], alpha=0.85, edgecolor='white', linewidth=0.4)
ov_med = ov_stats['50%']
ax2.axvline(ov_med, color=PALETTE[1], linewidth=1.8, linestyle='--', zorder=5)
ylim2 = ax2.get_ylim()
ax2.annotate(
    f"Median = £{ov_med:.0f}",
    xy=(ov_med, ylim2[1] * 0.70),
    xytext=(ov_med + ov_clip_threshold * 0.05, ylim2[1] * 0.70),
    fontsize=9, color=PALETTE[1],
    arrowprops=dict(arrowstyle='->', color=PALETTE[1], lw=1.2),
)
fig2.suptitle("Order Value Distribution", fontweight='bold', fontsize=13)
ax2.set_title(f"Revenue per invoice — clipped at 99th pct (£{ov_clip_threshold:,.0f})", fontsize=10, color='dimgray')
ax2.set_xlabel("Order Value (£)", fontsize=10)
ax2.set_ylabel("Number of Orders", fontsize=10)

# ── Figure 3: Monthly Revenue Trend ──────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 5), constrained_layout=True)
fig3.set_dpi(200)
fig3.set_facecolor('white')
ax3.set_facecolor('white')

months_str = [str(p) for p in monthly_rev.index]
x_pos = range(len(months_str))
bars = ax3.bar(x_pos, monthly_rev.values / 1_000, color=PALETTE[0], alpha=0.85, width=0.75)

# Rotate and set tick labels manually (avoid FuncFormatter)
ax3.set_xticks(list(x_pos))
ax3.set_xticklabels(months_str, rotation=45, ha='right', fontsize=8)

# Y-axis: pre-compute tick labels in £k
max_rev_k = (monthly_rev.values / 1_000).max()
yticks = np.linspace(0, max_rev_k * 1.1, 6)
ax3.set_yticks(yticks)
ax3.set_yticklabels([f"£{v:,.0f}k" for v in yticks])

# Annotate max bar
peak_idx  = int(np.argmax(monthly_rev.values))
peak_val_k = monthly_rev.values[peak_idx] / 1_000
ax3.annotate(
    f"Peak\n£{peak_val_k:,.0f}k",
    xy=(peak_idx, peak_val_k),
    xytext=(peak_idx + 1.2, peak_val_k * 0.92),
    fontsize=8.5, color='#333333',
    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#cccccc'),
)
fig3.suptitle("Monthly Revenue Trend", fontweight='bold', fontsize=13)
ax3.set_title("Total revenue (£k) per calendar month", fontsize=10, color='dimgray')
ax3.set_xlabel("Month", fontsize=10)
ax3.set_ylabel("Revenue (£ thousands)", fontsize=10)

# ── Figure 4: Inter-Purchase Gap Distribution ─────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 5), constrained_layout=True)
fig4.set_dpi(200)
fig4.set_facecolor('white')
ax4.set_facecolor('white')

gap_clip = gap_series[gap_series <= gap_series.quantile(0.99)]
ax4.hist(gap_clip.values, bins=80, color=PALETTE[4], alpha=0.85, edgecolor='white', linewidth=0.4)
ax4.axvline(gap_med, color=PALETTE[1], linewidth=1.8, linestyle='--', zorder=5)
ylim4 = ax4.get_ylim()
ax4.annotate(
    f"Median = {gap_med:.0f} days",
    xy=(gap_med, ylim4[1] * 0.70),
    xytext=(gap_med + gap_series.quantile(0.99) * 0.06, ylim4[1] * 0.70),
    fontsize=9, color=PALETTE[1],
    arrowprops=dict(arrowstyle='->', color=PALETTE[1], lw=1.2),
)
ax4.axvline(gap_p75 * 2, color='#9E4A4A', linewidth=1.4, linestyle=':', zorder=5, label=f"Churn threshold ≈ {int(gap_p75*2)} days (2×P75)")
ax4.legend(fontsize=9, framealpha=0.7, edgecolor='#cccccc')
fig4.suptitle("Inter-Purchase Gap Distribution", fontweight='bold', fontsize=13)
ax4.set_title("Days between consecutive orders (same customer, clipped at 99th pct)", fontsize=9, color='dimgray')
ax4.set_xlabel("Gap (days)", fontsize=10)
ax4.set_ylabel("Number of Purchase Pairs", fontsize=10)

# ── Assign named figure variables before closing ──────────────────────────────
fig_order_frequency  = fig1
fig_order_value      = fig2
fig_monthly_revenue  = fig3
fig_interpurchase_gap = fig4

plt.close('all')

print("\n  ✓ All 4 charts generated successfully.")
print(f"{'='*65}")
