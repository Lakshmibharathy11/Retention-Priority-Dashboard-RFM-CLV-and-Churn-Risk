import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Step 3 — Data Cleaning & RFM Analysis ────────────────────────────────────

REFERENCE_DATE = pd.Timestamp('2011-12-10')

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
n_before = len(df_raw)

# Apply all filters sequentially with counts at each step
_df = df_raw.copy()

# Remove null Customer ID
_mask_null_cust = _df[CUSTOMER_COL].isnull()
_n_null_cust = _mask_null_cust.sum()
_df = _df[~_mask_null_cust]

# Remove exact duplicate rows
_n_dupes = _df.duplicated().sum()
_df = _df.drop_duplicates()

# Remove Quantity <= 0
_mask_qty = _df[QTY_COL] <= 0
_n_qty = _mask_qty.sum()
_df = _df[~_mask_qty]

# Remove Price <= 0
_mask_price = _df[PRICE_COL] <= 0
_n_price = _mask_price.sum()
_df = _df[~_mask_price]

# Remove cancellation invoices (Invoice starts with 'C')
_mask_cancel = _df[INVOICE_COL].astype(str).str.startswith('C')
_n_cancel = _mask_cancel.sum()
_df = _df[~_mask_cancel]

# Working dataset for all further steps
df_clean_rfm = _df.reset_index(drop=True)
n_after = len(df_clean_rfm)
n_removed = n_before - n_after

print("=" * 65)
print("  DATA CLEANING SUMMARY")
print("=" * 65)
print(f"\n  {'Step':<45} {'Rows Removed':>12}")
print(f"  {'-'*45} {'-'*12}")
print(f"  {'Null Customer ID':<45} {_n_null_cust:>12,}")
print(f"  {'Exact duplicate rows':<45} {_n_dupes:>12,}")
print(f"  {'Quantity <= 0':<45} {_n_qty:>12,}")
print(f"  {'Price <= 0':<45} {_n_price:>12,}")
print(f"  {'Cancellation invoices (Invoice starts with C)':<45} {_n_cancel:>12,}")
print(f"  {'-'*45} {'-'*12}")
print(f"  {'Total rows removed':<45} {n_removed:>12,}")
print(f"\n  Row count BEFORE cleaning : {n_before:>12,}")
print(f"  Row count AFTER  cleaning : {n_after:>12,}")
print(f"  Retention rate            : {n_after/n_before*100:>11.1f}%")
print(f"\n  Working dataset saved as: df_clean_rfm")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE RFM METRICS PER CUSTOMER
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  COMPUTING RFM METRICS")
print(f"{'─'*65}")
print(f"  Reference date: {REFERENCE_DATE.date()}")

# Revenue column derived inline
_df_work = df_clean_rfm.copy()
_df_work['_revenue'] = _df_work[QTY_COL] * _df_work[PRICE_COL]

rfm = (
    _df_work
    .groupby(CUSTOMER_COL)
    .agg(
        Recency   = (DATE_COL,   lambda x: (REFERENCE_DATE - x.max()).days),
        Frequency = (INVOICE_COL, 'nunique'),
        Monetary  = ('_revenue', 'sum'),
    )
    .reset_index()
    .rename(columns={CUSTOMER_COL: 'CustomerID'})
)

print(f"\n  Customers in RFM table : {len(rfm):,}")
print(f"  Recency   — min: {rfm['Recency'].min():>4}d  max: {rfm['Recency'].max():>4}d  mean: {rfm['Recency'].mean():.1f}d")
print(f"  Frequency — min: {rfm['Frequency'].min():>4}   max: {rfm['Frequency'].max():>4}   mean: {rfm['Frequency'].mean():.1f}")
print(f"  Monetary  — min: £{rfm['Monetary'].min():>10,.2f}  max: £{rfm['Monetary'].max():>10,.2f}  mean: £{rfm['Monetary'].mean():,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. QUARTILE SCORING (1 = lowest, 4 = highest)
# ─────────────────────────────────────────────────────────────────────────────
# Recency: lower days = better = higher score → reverse rank
rfm['R_Score'] = pd.qcut(rfm['Recency'],   q=4, labels=[4, 3, 2, 1]).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'),  q=4, labels=[1, 2, 3, 4]).astype(int)

rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SEGMENT LABELLING
# ─────────────────────────────────────────────────────────────────────────────
def _assign_segment(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    rf = r * 10 + f  # convenience composite

    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'Recent Customers'
    elif r >= 3 and f >= 1 and m >= 3:
        return 'Potential Loyalists'
    elif r == 2 and f >= 3 and m >= 3:
        return 'At Risk'
    elif r == 2 and f <= 2:
        return 'Needs Attention'
    elif r == 1 and f >= 3 and m >= 3:
        return 'Lost – High Value'
    else:
        return 'Lost'

rfm['Segment'] = rfm.apply(_assign_segment, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. SEGMENT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
rfm_segment_summary = (
    rfm.groupby('Segment')
    .agg(
        Customers      = ('CustomerID',  'count'),
        Avg_Recency    = ('Recency',     'mean'),
        Avg_Frequency  = ('Frequency',   'mean'),
        Avg_Monetary   = ('Monetary',    'mean'),
    )
    .reset_index()
    .sort_values('Customers', ascending=False)
)

rfm_segment_summary['Avg_Recency']   = rfm_segment_summary['Avg_Recency'].round(1)
rfm_segment_summary['Avg_Frequency'] = rfm_segment_summary['Avg_Frequency'].round(1)
rfm_segment_summary['Avg_Monetary']  = rfm_segment_summary['Avg_Monetary'].round(2)

# Pretty-print the table
_col_w = [22, 10, 14, 15, 14]
_header = f"  {'Segment':<{_col_w[0]}} {'Customers':>{_col_w[1]}} {'Avg Recency':>{_col_w[2]}} {'Avg Frequency':>{_col_w[3]}} {'Avg Monetary':>{_col_w[4]}}"
_sep    = f"  {'-'*_col_w[0]} {'-'*_col_w[1]} {'-'*_col_w[2]} {'-'*_col_w[3]} {'-'*_col_w[4]}"

print(f"\n{'─'*65}")
print("  RFM SEGMENT SUMMARY")
print(f"{'─'*65}")
print(_header)
print(_sep)
for _, row in rfm_segment_summary.iterrows():
    print(
        f"  {row['Segment']:<{_col_w[0]}} "
        f"{row['Customers']:>{_col_w[1]},} "
        f"{row['Avg_Recency']:>{_col_w[2]}.1f} "
        f"{row['Avg_Frequency']:>{_col_w[3]}.1f} "
        f"£{row['Avg_Monetary']:>{_col_w[4]-1},.2f}"
    )
print(f"  {'-'*65}")
print(f"  {'TOTAL':<{_col_w[0]}} {rfm_segment_summary['Customers'].sum():>{_col_w[1]},}")
print(f"{'─'*65}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
_PALETTE = {
    'Champions'          : '#2ecc71',
    'Loyal Customers'    : '#27ae60',
    'Potential Loyalists': '#3498db',
    'Recent Customers'   : '#1abc9c',
    'Needs Attention'    : '#f39c12',
    'At Risk'            : '#e67e22',
    'Lost – High Value'  : '#e74c3c',
    'Lost'               : '#95a5a6',
}

_seg_order = rfm_segment_summary['Segment'].tolist()
_colors    = [_PALETTE.get(s, '#bdc3c7') for s in _seg_order]
_counts    = rfm_segment_summary['Customers'].tolist()

# — Fig A: Customer count per segment (horizontal bar) ——————————————————————
fig_rfm_segments, _ax = plt.subplots(figsize=(9, 5))
_bars = _ax.barh(_seg_order[::-1], _counts[::-1], color=_colors[::-1], edgecolor='white', linewidth=0.6)
for _bar, _val in zip(_bars, _counts[::-1]):
    _ax.text(_bar.get_width() + 12, _bar.get_y() + _bar.get_height() / 2,
             f'{_val:,}', va='center', ha='left', fontsize=9, color='#2c3e50')
_ax.set_xlabel('Number of Customers', fontsize=10)
_ax.set_title('Customer Count by RFM Segment', fontsize=13, fontweight='bold', pad=12)
_ax.spines[['top', 'right', 'left']].set_visible(False)
_ax.tick_params(axis='y', length=0)
_ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
_ax.set_xlim(0, max(_counts) * 1.18)
plt.tight_layout()
plt.close('all')

# — Fig B: Avg Monetary per segment ————————————————————————————————————————
_mon_vals  = rfm_segment_summary['Avg_Monetary'].tolist()
fig_rfm_monetary, _ax2 = plt.subplots(figsize=(9, 5))
_bars2 = _ax2.barh(_seg_order[::-1], _mon_vals[::-1], color=_colors[::-1], edgecolor='white', linewidth=0.6)
for _bar2, _val2 in zip(_bars2, _mon_vals[::-1]):
    _ax2.text(_bar2.get_width() + 30, _bar2.get_y() + _bar2.get_height() / 2,
              f'£{_val2:,.0f}', va='center', ha='left', fontsize=9, color='#2c3e50')
_ax2.set_xlabel('Average Revenue (£)', fontsize=10)
_ax2.set_title('Average Monetary Value by RFM Segment', fontsize=13, fontweight='bold', pad=12)
_ax2.spines[['top', 'right', 'left']].set_visible(False)
_ax2.tick_params(axis='y', length=0)
_ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{int(x):,}'))
_ax2.set_xlim(0, max(_mon_vals) * 1.22)
plt.tight_layout()
plt.close('all')

# — Fig C: RFM scatter — Frequency vs Recency, sized by Monetary ————————————
fig_rfm_scatter, _ax3 = plt.subplots(figsize=(9, 6))
for _seg in _seg_order:
    _sub = rfm[rfm['Segment'] == _seg]
    _sz  = np.clip(_sub['Monetary'] / _sub['Monetary'].max() * 300, 10, 300)
    _ax3.scatter(
        _sub['Recency'], _sub['Frequency'],
        s=_sz, alpha=0.55, label=_seg,
        color=_PALETTE.get(_seg, '#bdc3c7'), edgecolors='white', linewidths=0.4
    )
_ax3.set_xlabel('Recency (days since last purchase)', fontsize=10)
_ax3.set_ylabel('Frequency (distinct invoices)', fontsize=10)
_ax3.set_title('RFM Customer Map — Frequency vs Recency\n(bubble size ∝ Monetary value)', fontsize=12, fontweight='bold', pad=10)
_ax3.legend(fontsize=8, framealpha=0.7, loc='upper right')
_ax3.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.close('all')

print("\n  Outputs: rfm (customer-level), rfm_segment_summary (segment table)")
print("  Charts:  fig_rfm_segments | fig_rfm_monetary | fig_rfm_scatter")