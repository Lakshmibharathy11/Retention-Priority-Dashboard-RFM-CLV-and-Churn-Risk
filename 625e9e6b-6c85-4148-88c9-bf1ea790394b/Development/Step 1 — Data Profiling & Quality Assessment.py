
# ── Step 1 — Data Profiling & Quality Assessment ───────────────────────────
# Structured assessment of the raw dataset. Raw dataframe is NOT modified.

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. SHAPE
# ─────────────────────────────────────────────────────────────────────────────
n_rows, n_cols = df_raw.shape
print("=" * 65)
print("  UCI ONLINE RETAIL II — DATA PROFILING REPORT")
print("=" * 65)
print(f"\n{'─'*65}")
print(f"  1. SHAPE")
print(f"{'─'*65}")
print(f"  Total rows    : {n_rows:>12,}")
print(f"  Total columns : {n_cols:>12,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  2. DATA TYPES")
print(f"{'─'*65}")
for col, dtype in df_raw.dtypes.items():
    print(f"  {col:<20} {str(dtype):<15}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  3. MISSING VALUES")
print(f"{'─'*65}")
null_counts = df_raw.isnull().sum()
null_pct = (null_counts / n_rows * 100).round(2)
print(f"  {'Column':<22} {'Missing Count':>14} {'Missing %':>10}")
print(f"  {'-'*22} {'-'*14} {'-'*10}")
for col in df_raw.columns:
    print(f"  {col:<22} {null_counts[col]:>14,} {null_pct[col]:>9.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DUPLICATE ROWS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  4. DUPLICATE ROWS (all columns identical)")
print(f"{'─'*65}")
n_dupes = df_raw.duplicated().sum()
print(f"  Exact duplicate rows: {n_dupes:>10,}  ({n_dupes/n_rows*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. INVALID RECORDS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  5. INVALID / FLAGGED RECORDS")
print(f"{'─'*65}")

mask_qty_le0      = df_raw[QTY_COL] <= 0
mask_price_le0    = df_raw[PRICE_COL] <= 0
mask_no_customer  = df_raw[CUSTOMER_COL].isnull()
mask_cancellation = df_raw[INVOICE_COL].astype(str).str.startswith('C')

n_qty_le0      = mask_qty_le0.sum()
n_price_le0    = mask_price_le0.sum()
n_no_customer  = mask_no_customer.sum()
n_cancellation = mask_cancellation.sum()

print(f"  {'Issue':<42} {'Count':>10} {'% of Total':>11}")
print(f"  {'-'*42} {'-'*10} {'-'*11}")
print(f"  {'Quantity <= 0 (returns/anomalies)':<42} {n_qty_le0:>10,} {n_qty_le0/n_rows*100:>10.2f}%")
print(f"  {'Price <= 0':<42} {n_price_le0:>10,} {n_price_le0/n_rows*100:>10.2f}%")
print(f"  {'Customer ID is null':<42} {n_no_customer:>10,} {n_no_customer/n_rows*100:>10.2f}%")
print(f"  {'Invoice starts with C (cancellations)':<42} {n_cancellation:>10,} {n_cancellation/n_rows*100:>10.2f}%")

# Count rows that are ANY of the above flags
mask_any_issue = mask_qty_le0 | mask_price_le0 | mask_no_customer | mask_cancellation
n_any_issue = mask_any_issue.sum()
print(f"\n  Rows flagged for ANY issue (union): {n_any_issue:>10,}  ({n_any_issue/n_rows*100:.2f}%)")
print(f"  Clean rows (no issues):             {n_rows - n_any_issue:>10,}  ({(n_rows-n_any_issue)/n_rows*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CONSOLIDATED SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  6. CONSOLIDATED PROFILING SUMMARY")
print(f"{'─'*65}")

summary = {
    'Total rows'                        : f"{n_rows:,}",
    'Total columns'                     : f"{n_cols}",
    'Exact duplicate rows'              : f"{n_dupes:,} ({n_dupes/n_rows*100:.2f}%)",
    'Rows with null Customer ID'        : f"{n_no_customer:,} ({n_no_customer/n_rows*100:.2f}%)",
    'Rows with null Description'        : f"{null_counts.get('Description',0):,} ({null_counts.get('Description',0)/n_rows*100:.2f}%)",
    'Rows with Quantity <= 0'           : f"{n_qty_le0:,} ({n_qty_le0/n_rows*100:.2f}%)",
    'Rows with Price <= 0'              : f"{n_price_le0:,} ({n_price_le0/n_rows*100:.2f}%)",
    'Cancellation rows (Invoice ~ C*)' : f"{n_cancellation:,} ({n_cancellation/n_rows*100:.2f}%)",
    'Rows flagged (any issue)'          : f"{n_any_issue:,} ({n_any_issue/n_rows*100:.2f}%)",
    'Rows clean (no issues)'            : f"{n_rows-n_any_issue:,} ({(n_rows-n_any_issue)/n_rows*100:.2f}%)",
    'Date range (raw)'                  : f"{df_raw[DATE_COL].min().date()} → {df_raw[DATE_COL].max().date()}",
}

for metric, value in summary.items():
    print(f"  {metric:<42} {value}")

print(f"\n{'='*65}")
print("  Raw dataframe preserved as df_raw (no rows dropped).")
print(f"{'='*65}")

# Export flag masks for potential downstream use
profiling_flags = {
    'mask_qty_le0'     : mask_qty_le0,
    'mask_price_le0'   : mask_price_le0,
    'mask_no_customer' : mask_no_customer,
    'mask_cancellation': mask_cancellation,
    'mask_any_issue'   : mask_any_issue,
}
