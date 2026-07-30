
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

# ── Configuration ──────────────────────────────────────────────────────────────
TOP_N              = 50
CLV_COL            = 'clv'                   # actual column name in rfm_clv
CHURN_PROB_COL     = 'Churn_Probability'
DISPLAY_COLS       = ['CustomerID', 'Segment', 'Recency', 'Frequency',
                      'Monetary', 'clv', 'Churn_Probability', 'Expected_Revenue_At_Risk']
DISPLAY_HEADERS    = ['CustomerID', 'Segment', 'Recency', 'Frequency',
                      'Monetary', 'CLV', 'Churn_Probability', 'Expected_Revenue_At_Risk']
PALETTE            = ['#3D5A80', '#B07D62', '#6B8F71', '#8C6A9E',
                      '#C9A227', '#4C8C8C', '#9E4A4A', '#5A5A5A']

# ── Step 1 — Expected Revenue at Risk ─────────────────────────────────────────
rfm_clv['Expected_Revenue_At_Risk'] = rfm_clv[CLV_COL] * rfm_clv[CHURN_PROB_COL]

# ── Step 2 — Build retention priority list ─────────────────────────────────────
# Ensure CustomerID is a regular column (not buried in index)
rfm_work = rfm_clv.copy()
if 'CustomerID' not in rfm_work.columns and rfm_work.index.name in ('CustomerID', 'Customer ID'):
    rfm_work = rfm_work.reset_index().rename(columns={rfm_work.index.name: 'CustomerID'})

# Sort by Expected_Revenue_At_Risk descending and take top 50
retention_priority = (rfm_work
                      .sort_values('Expected_Revenue_At_Risk', ascending=False)
                      .head(TOP_N)
                      .reset_index(drop=True))

# ── Print formatted table ──────────────────────────────────────────────────────
pd.set_option('display.max_rows', 60)
pd.set_option('display.float_format', '{:,.2f}'.format)

display_df = retention_priority[DISPLAY_COLS].copy()
display_df.columns = DISPLAY_HEADERS          # rename clv → CLV for display

display_fmt = display_df.copy()
for col in ['Monetary', 'CLV', 'Expected_Revenue_At_Risk']:
    display_fmt[col] = display_fmt[col].map('{:,.2f}'.format)
display_fmt['Churn_Probability'] = display_fmt['Churn_Probability'].map('{:.4f}'.format)

print("=" * 110)
print(f"  TOP-{TOP_N} RETENTION PRIORITY LIST  (sorted by Expected Revenue at Risk, descending)")
print("=" * 110)
print(display_fmt.to_string(index=True))

# ── Step 3 — CLV concentration stats ──────────────────────────────────────────
top50_clv_sum = retention_priority[CLV_COL].sum()
total_clv_sum = rfm_work[CLV_COL].sum()
top50_clv_pct = top50_clv_sum / total_clv_sum * 100

print()
print("─" * 55)
print("  CLV CONCENTRATION STATS")
print("─" * 55)
print(f"  Top-50 CLV sum      : £{top50_clv_sum:,.0f}")
print(f"  Total CLV (all)     : £{total_clv_sum:,.0f}")
print(f"  Top-50 share of CLV : {top50_clv_pct:.1f}%")
print("─" * 55)

# ── Step 4 — Segment breakdown of top 50 ──────────────────────────────────────
top50_segment_breakdown = (retention_priority['Segment']
                           .value_counts()
                           .sort_values(ascending=False))

print()
print("─" * 55)
print("  SEGMENT BREAKDOWN — TOP 50")
print("─" * 55)
for seg, cnt in top50_segment_breakdown.items():
    print(f"  {seg:<32s} : {cnt}")
print("─" * 55)

# ── Validation ─────────────────────────────────────────────────────────────────
assert len(retention_priority) == TOP_N, f"Expected {TOP_N} rows, got {len(retention_priority)}"
print(f"\n✓ retention_priority has {len(retention_priority)} rows (correct)")
print(f"  Expected_Revenue_At_Risk sample check: {retention_priority['Expected_Revenue_At_Risk'].iloc[0]:,.2f} "
      f"= {retention_priority[CLV_COL].iloc[0]:,.2f} × {retention_priority[CHURN_PROB_COL].iloc[0]:.4f} ✓")

# ── Step 5 — Charts ────────────────────────────────────────────────────────────

# Chart 1: Horizontal bar — segment breakdown
segments   = top50_segment_breakdown.index.tolist()
counts     = top50_segment_breakdown.values.tolist()
n_segs     = len(segments)
bar_colors = (PALETTE * ((n_segs // len(PALETTE)) + 1))[:n_segs]

fig1, ax1 = plt.subplots(figsize=(9, max(4, 0.58 * n_segs + 1.8)), constrained_layout=True,
                          facecolor='white')
ax1.set_facecolor('white')
fig1.set_dpi(200)

bars_h = ax1.barh(segments[::-1], counts[::-1], color=bar_colors[::-1],
                  edgecolor='none', height=0.6)
for bobj, cnt in zip(bars_h, counts[::-1]):
    ax1.text(bobj.get_width() + 0.15, bobj.get_y() + bobj.get_height() / 2,
             str(cnt), va='center', ha='left', fontsize=10, color='#333333')

ax1.spines[['top', 'right', 'left']].set_visible(False)
ax1.xaxis.grid(True, color='#E0E0E0', linewidth=0.7)
ax1.set_axisbelow(True)
ax1.set_xlabel('Number of Customers', fontsize=11)
ax1.set_xlim(0, max(counts) * 1.22)
ax1.tick_params(axis='y', length=0, labelsize=10)
ax1.tick_params(axis='x', labelsize=9)
fig1.suptitle("Top-50 Retention Priority: Segment Breakdown", fontweight='bold', fontsize=13)

fig_retention_segment_breakdown = fig1
plt.close('all')

# Chart 2: Scatter — Expected Revenue at Risk vs CLV
top50_ids = set(retention_priority['CustomerID'].tolist())
is_top50  = rfm_work['CustomerID'].isin(top50_ids)
rest_df   = rfm_work[~is_top50]
top_df    = rfm_work[is_top50]

fig2, ax2 = plt.subplots(figsize=(10, 7), constrained_layout=True, facecolor='white')
ax2.set_facecolor('white')
fig2.set_dpi(200)

ax2.scatter(rest_df[CLV_COL], rest_df['Expected_Revenue_At_Risk'],
            color='#AABBD0', alpha=0.45, s=18, linewidths=0,
            label='All other customers', zorder=2)
ax2.scatter(top_df[CLV_COL], top_df['Expected_Revenue_At_Risk'],
            color='#B07D62', alpha=0.90, s=55, linewidths=0.5, edgecolors='white',
            label='Top 50 at-risk', zorder=3)

ax2.spines[['top', 'right']].set_visible(False)
ax2.xaxis.grid(True, color='#E8E8E8', linewidth=0.6)
ax2.yaxis.grid(True, color='#E8E8E8', linewidth=0.6)
ax2.set_axisbelow(True)
ax2.set_xlabel('CLV (£)', fontsize=12)
ax2.set_ylabel('Expected Revenue at Risk (£)', fontsize=12)
ax2.tick_params(labelsize=10)

# Pre-compute tick labels without callable formatters
raw_xticks = ax2.get_xticks()
xlabels = [f'£{v/1000:.0f}K' if abs(v) >= 1000 else f'£{v:.0f}' for v in raw_xticks]
ax2.set_xticklabels(xlabels, fontsize=9)

raw_yticks = ax2.get_yticks()
ylabels = [f'£{v/1000:.0f}K' if abs(v) >= 1000 else f'£{v:.0f}' for v in raw_yticks]
ax2.set_yticklabels(ylabels, fontsize=9)

ax2.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC', fontsize=10, loc='upper left')
ax2.set_title(f"n = {len(rfm_work):,} customers · Top 50 highlighted in clay",
              fontsize=10, color='dimgray')
fig2.suptitle("Expected Revenue at Risk vs CLV — All Customers",
              fontweight='bold', fontsize=13)

fig_retention_scatter = fig2
plt.close('all')

print("\n✓ Charts rendered: fig_retention_segment_breakdown, fig_retention_scatter")
