
# ── Data Load ──────────────────────────────────────────────────────────────
# Load the UCI Online Retail II dataset and define shared constants.

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ───────────────────────────────────────────────────────────
DATA_PATH       = './online_retail_II.csv'
REVENUE_COL     = 'LineRevenue'
CUSTOMER_COL    = 'Customer ID'
INVOICE_COL     = 'Invoice'
DATE_COL        = 'InvoiceDate'
QTY_COL         = 'Quantity'
PRICE_COL       = 'Price'
STOCK_COL       = 'StockCode'

# ── Load ─────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(
    DATA_PATH,
    encoding='utf-8',
    on_bad_lines='skip',
    dtype={CUSTOMER_COL: str},   # keep as string to preserve NaN representation
)

# Replace literal 'nan' strings with actual NaN
df_raw[CUSTOMER_COL] = df_raw[CUSTOMER_COL].replace('nan', np.nan)

# Parse InvoiceDate – the UCI dataset actually stores as ISO-style "YYYY-MM-DD HH:MM:SS"
# Use format='mixed' with dayfirst=True to handle any variation
df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL], format='mixed', dayfirst=True)

print(f"✓ Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns from {DATA_PATH}")
print(f"  Columns: {list(df_raw.columns)}")
print(f"  Date range (raw): {df_raw[DATE_COL].min()} → {df_raw[DATE_COL].max()}")
print(f"  Sample dates: {df_raw[DATE_COL].head(3).tolist()}")
