import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── CLV Calculation ───────────────────────────────────────────────────────────
# Churn threshold from earlier analysis
CHURN_THRESHOLD_DAYS = 138
DATASET_SPAN_YEARS   = 2.9   # conservative fallback for single-purchase customers

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHURN RATE & AVERAGE CUSTOMER LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────
churn_rate       = (rfm['Recency'] > CHURN_THRESHOLD_DAYS).mean()
avg_lifespan_yrs = 1 / churn_rate

print("=" * 60)
print("  CLV INPUTS")
print("=" * 60)
print(f"  Churn threshold        : {CHURN_THRESHOLD_DAYS} days")
print(f"  Churned customers      : {(rfm['Recency'] > CHURN_THRESHOLD_DAYS).sum():,} / {len(rfm):,}")
print(f"  Churn rate             : {churn_rate:.4f}  ({churn_rate*100:.2f}%)")
print(f"  Avg customer lifespan  : {avg_lifespan_yrs:.2f} years")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CUSTOMER-LEVEL TENURE (first → last purchase in df_clean_rfm)
# ─────────────────────────────────────────────────────────────────────────────
_tenure_df = (
    df_clean_rfm
    .groupby(CUSTOMER_COL)[DATE_COL]
    .agg(first_purchase='min', last_purchase='max')
    .reset_index()
    .rename(columns={CUSTOMER_COL: 'CustomerID'})
)
_tenure_df['tenure_days']  = (_tenure_df['last_purchase'] - _tenure_df['first_purchase']).dt.days
_tenure_df['tenure_years'] = _tenure_df['tenure_days'] / 365.25

# For single-purchase customers (tenure == 0), use dataset span as conservative estimate
_tenure_df['tenure_years'] = _tenure_df['tenure_years'].where(
    _tenure_df['tenure_years'] > 0, other=DATASET_SPAN_YEARS
)

_n_single = (_tenure_df['tenure_days'] == 0).sum()
print(f"\n  Single-purchase customers (using {DATASET_SPAN_YEARS}yr fallback): {_n_single:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MERGE TENURE INTO RFM & COMPUTE CLV COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
rfm_clv = rfm.merge(_tenure_df[['CustomerID', 'tenure_years']], on='CustomerID', how='left')

# Average Order Value = total revenue / number of orders
rfm_clv['avg_order_value']       = rfm_clv['Monetary'] / rfm_clv['Frequency']

# Purchase Frequency Rate = orders per year
rfm_clv['purchase_freq_rate']    = rfm_clv['Frequency'] / rfm_clv['tenure_years']

# CLV = AOV × Purchase Frequency Rate × Avg Lifespan
rfm_clv['clv']                   = (
    rfm_clv['avg_order_value']
    * rfm_clv['purchase_freq_rate']
    * avg_lifespan_yrs
)

print(f"\n  CLV computed for {len(rfm_clv):,} customers")
print(f"  CLV range: £{rfm_clv['clv'].min():,.2f} – £{rfm_clv['clv'].max():,.2f}")
print(f"  Median CLV: £{rfm_clv['clv'].median():,.2f}")
print(f"  Mean CLV  : £{rfm_clv['clv'].mean():,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SEGMENT SUMMARY — AVG CLV PER SEGMENT (sorted highest → lowest)
# ─────────────────────────────────────────────────────────────────────────────
clv_segment_summary = (
    rfm_clv.groupby('Segment')
    .agg(
        Customers          = ('CustomerID',        'count'),
        Avg_CLV            = ('clv',               'mean'),
        Median_CLV         = ('clv',               'median'),
        Avg_AOV            = ('avg_order_value',   'mean'),
        Avg_Freq_Rate      = ('purchase_freq_rate','mean'),
    )
    .reset_index()
    .sort_values('Avg_CLV', ascending=False)
    .reset_index(drop=True)
)

# Pretty-print the table
_cw = [22, 10, 14, 14, 12, 16]
_hdr = (
    f"  {'Segment':<{_cw[0]}} {'Customers':>{_cw[1]}} "
    f"{'Avg CLV':>{_cw[2]}} {'Median CLV':>{_cw[3]}} "
    f"{'Avg AOV':>{_cw[4]}} {'Freq/yr':>{_cw[5]}}"
)
_sep = f"  {'-'*_cw[0]} {'-'*_cw[1]} {'-'*_cw[2]} {'-'*_cw[3]} {'-'*_cw[4]} {'-'*_cw[5]}"

print(f"\n{'─'*90}")
print("  CLV BY RFM SEGMENT  (sorted highest → lowest Avg CLV)")
print(f"{'─'*90}")
print(_hdr)
print(_sep)
for _rank, (_idx, _row) in enumerate(clv_segment_summary.iterrows(), start=1):
    print(
        f"  {_rank}. {_row['Segment']:<{_cw[0]-3}} "
        f"{_row['Customers']:>{_cw[1]},} "
        f"£{_row['Avg_CLV']:>{_cw[2]-1},.0f} "
        f"£{_row['Median_CLV']:>{_cw[3]-1},.0f} "
        f"£{_row['Avg_AOV']:>{_cw[4]-1},.0f} "
        f"{_row['Avg_Freq_Rate']:>{_cw[5]}.2f}"
    )
print(f"{'─'*90}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATION — Avg CLV by segment (horizontal bar)
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

_seg_order_clv = clv_segment_summary['Segment'].tolist()
_clv_vals      = clv_segment_summary['Avg_CLV'].tolist()
_colors_clv    = [_PALETTE.get(s, '#bdc3c7') for s in _seg_order_clv]

fig_clv_segment, _ax_clv = plt.subplots(figsize=(10, 5))
_bars_clv = _ax_clv.barh(
    _seg_order_clv[::-1], _clv_vals[::-1],
    color=_colors_clv[::-1], edgecolor='white', linewidth=0.6
)
for _b, _v in zip(_bars_clv, _clv_vals[::-1]):
    _ax_clv.text(
        _b.get_width() + max(_clv_vals) * 0.015,
        _b.get_y() + _b.get_height() / 2,
        f'£{_v:,.0f}', va='center', ha='left', fontsize=9, color='#2c3e50'
    )
_ax_clv.set_xlabel('Average CLV (£)', fontsize=10)
_ax_clv.set_title(
    'Average Customer Lifetime Value by RFM Segment',
    fontsize=13, fontweight='bold', pad=12
)
_ax_clv.spines[['top', 'right', 'left']].set_visible(False)
_ax_clv.tick_params(axis='y', length=0)
_ax_clv.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{int(x):,}'))
_ax_clv.set_xlim(0, max(_clv_vals) * 1.22)
plt.tight_layout()
plt.close('all')

print(f"\n  Output: rfm_clv (customer-level with CLV), clv_segment_summary (segment table)")
print(f"  Chart : fig_clv_segment")