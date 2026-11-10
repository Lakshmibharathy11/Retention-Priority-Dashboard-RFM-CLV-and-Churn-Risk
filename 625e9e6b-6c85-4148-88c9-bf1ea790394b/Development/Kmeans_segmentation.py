"""K-means customer segmentation on RFM features (UCI Online Retail II).

Extends the rule-based RFM segments with an unsupervised view.
Steps: clean data (same rules as rfm_analysis_segmentation.py), build RFM,
log-transform + standardize, choose k with silhouette and Davies-Bouldin scores,
check stability across random seeds, profile clusters, compare to RFM segments.
"""
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

DATA_PATH = './online_retail_II.xlsx'          # UCI Online Retail II (both sheets)
REFERENCE_DATE = pd.Timestamp('2011-12-10')
CHURN_DAYS = 138                                # same rule as the churn model

# 1. Load + clean (mirrors rfm_analysis_segmentation.py)
sheets = pd.read_excel(DATA_PATH, sheet_name=None, dtype={'Customer ID': str})
df = pd.concat(sheets.values(), ignore_index=True)
df = df[df['Customer ID'].notna()].drop_duplicates()
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
df = df[~df['Invoice'].astype(str).str.startswith('C')]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['revenue'] = df['Quantity'] * df['Price']

# 2. RFM table
rfm = (df.groupby('Customer ID')
         .agg(Recency=('InvoiceDate', lambda x: (REFERENCE_DATE - x.max()).days),
              Frequency=('Invoice', 'nunique'),
              Monetary=('revenue', 'sum'))
         .reset_index())
print(f"Customers: {len(rfm):,}")

# 3. Feature transformation: RFM is heavily right-skewed, so log1p then standardize
X_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
print("Skew before:", rfm[['Recency', 'Frequency', 'Monetary']].skew().round(2).to_dict())
print("Skew after :", X_log.skew().round(2).to_dict())
X = StandardScaler().fit_transform(X_log)

# 4. Choose k
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
    rows.append((k, km.inertia_, silhouette_score(X, km.labels_), davies_bouldin_score(X, km.labels_)))
print(pd.DataFrame(rows, columns=['k', 'inertia', 'silhouette', 'davies_bouldin']).round(3).to_string(index=False))
# k=2 has the top silhouette but only splits active vs lapsed; k=4 is the best
# score among k>=3 (highest silhouette, lowest Davies-Bouldin) and is actionable.
K = 4

# 5. Fit + stability check across seeds
labels = [KMeans(n_clusters=K, n_init=20, random_state=s).fit(X).labels_ for s in range(5)]
print("Adjusted Rand vs seed 0:", [round(adjusted_rand_score(labels[0], l), 3) for l in labels[1:]])
rfm['Cluster'] = labels[0]

# 6. Profile clusters
total = rfm['Monetary'].sum()
profile = (rfm.groupby('Cluster')
              .agg(Customers=('Recency', 'size'),
                   Median_Recency=('Recency', 'median'),
                   Median_Frequency=('Frequency', 'median'),
                   Median_Monetary=('Monetary', 'median'),
                   Revenue=('Monetary', 'sum'),
                   Churn_Rate=('Recency', lambda r: (r > CHURN_DAYS).mean())))
profile['Pct_Customers'] = (profile['Customers'] / len(rfm) * 100).round(1)
profile['Pct_Revenue'] = (profile['Revenue'] / total * 100).round(1)
profile['Churn_Rate'] = (profile['Churn_Rate'] * 100).round(1)
print(profile.sort_values('Revenue', ascending=False).to_string())
# Note: Churn_Rate uses the Recency-based rule, so it describes clusters; it is
# not an independent validation (Recency is also a clustering feature).

rfm.to_csv('rfm_kmeans_clusters.csv', index=False)
