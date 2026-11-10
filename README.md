# Retention Priority Dashboard: RFM, CLV and Churn Risk

An end-to-end customer-retention analysis built on transactional retail data. The project turns individual purchases into a practical retention queue: **which customers are valuable, which are likely to leave, and where should a retention team focus first?**

It is written as a data-science case study, balancing business clarity with reproducible analytical work: data quality assessment, exploratory analysis, customer segmentation, customer lifetime value (CLV), churn modelling, model comparison, and prioritisation.

## Business problem

Retention teams often have limited time and budget. Treating every inactive customer in the same way can waste effort on low-value accounts while missing customers whose departure would have a meaningful revenue impact.

This project answers four questions:

1. What does healthy and unhealthy purchase behaviour look like?
2. Which customer groups are most valuable or most at risk?
3. Which customers have the highest predicted likelihood of churn?
4. How can churn risk and customer value be combined into an actionable contact list?

## What the analysis delivers

| Output | Business use |
| --- | --- |
| Data-quality report | Makes the scope and cleaning decisions transparent. |
| Exploratory data analysis | Describes order values, purchase frequency, revenue trends, purchase gaps, and revenue concentration. |
| RFM customer segments | Groups customers into understandable profiles such as Champions, Loyal Customers, At Risk, and Lost - High Value. |
| CLV estimates | Adds an estimated long-term value to each customer. |
| Churn probabilities | Flags customers whose purchase history resembles churned customers. |
| Top-50 retention list | Ranks customers by expected revenue at risk for immediate follow-up. |

## Dataset

The analysis uses the [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii), a two-year transaction history from a UK-based non-store retailer. The source dataset contains over one million transaction rows and includes invoice, product, quantity, price, customer, country, and timestamp fields.

The project expects a local CSV named `online_retail_II.csv` in the project root. Its scripts use these key fields:

`Invoice`, `InvoiceDate`, `Quantity`, `Price`, `Customer ID`, and `StockCode`.

> The source download is an Excel file and may use slightly different column labels (for example, `InvoiceNo`, `UnitPrice`, and `CustomerID`). Convert it to CSV and align the column names with the project configuration in [`Development/Data Load.py`](Development/Data%20Load.py).

## Analytical workflow

```text
Raw transactions
      |
      v
Profiling and cleaning
      |
      +--> exploratory analysis and churn-threshold research
      |
      v
Customer-level RFM table
      |
      +--> RFM segments
      +--> CLV estimates
      +--> churn model and probability
                         |
                         v
Expected revenue at risk = CLV x churn probability
                         |
                         v
Top-50 retention priority list
```

### 1. Data quality and preparation

The workflow profiles the raw data before modifying it. For the RFM and modelling dataset, it removes:

- transactions without a customer ID;
- duplicate rows;
- non-positive quantities or prices; and
- cancelled invoices (invoice IDs beginning with `C`).

Revenue is calculated at line level as `Quantity x Price`. These rules deliberately keep the analysis focused on completed, attributable purchases. The removal counts are printed so the impact of each decision is auditable.

### 2. Exploratory analysis and churn definition

The exploratory stage examines order frequency, order values, monthly revenue, customer concentration, and the time between repeat orders. It proposes a churn threshold based on actual customer behaviour:

`churn threshold = 2 x 75th percentile of inter-purchase gaps`

For this dataset, the threshold is **138 days**. A customer who has not bought in more than 138 days is labelled churned for the retrospective analysis. This is a transparent business rule, not a claim that churn is directly observed in the source data.

### 3. RFM segmentation

Each customer receives three measures using a reference date of 10 December 2011:

- **Recency** — days since their most recent purchase (lower is better).
- **Frequency** — number of distinct invoices (higher is better).
- **Monetary** — total completed-purchase revenue (higher is better).

Each measure is scored in quartiles from 1 to 4. The three scores are then translated into business-friendly segments: Champions, Loyal Customers, Potential Loyalists, Recent Customers, Needs Attention, At Risk, Lost - High Value, and Lost.

### 4. Customer lifetime value

RFM describes behaviour but not the financial value of retaining a customer. This project therefore estimates CLV using:

`CLV = average order value x purchase frequency per year x estimated average customer lifespan`

Customer tenure is calculated from first to last purchase. For customers with only one recorded purchase, the dataset span is used as a conservative fallback. The churn rate implied by the 138-day rule is used to estimate average lifespan.

### K-means customer segmentation (extension)

The rule-based RFM segments use fixed quartile cut-offs. To check whether the data itself groups customers differently, I added an unsupervised view (`kmeans_segmentation.py`).

**Method**
- Same cleaning rules as the RFM step: 1,067,371 raw rows → 779,425 clean rows → 5,878 customers.
- Feature transformation: RFM values are heavily right-skewed (Monetary skew 25.07, Frequency 12.64), so I applied `log1p` and standardized them. Skew after the log transform: Monetary 0.27, Frequency 1.00, Recency −0.55.
- Choosing k: tested k = 2 to 8 with silhouette and Davies-Bouldin scores. k = 2 scores highest (silhouette 0.439) but only separates active from lapsed customers. **k = 4** is the best among k ≥ 3 (silhouette 0.365, Davies-Bouldin 0.941) and gives actionable groups.
- Stability: cluster labels are nearly identical across 5 random seeds (Adjusted Rand Index ≥ 0.996).

**Clusters (k = 4)**

| Cluster | Customers | Revenue share | Median recency | Median orders | Churned (>138 days) |
|---|---|---|---|---|---|
| High value, active | 19.0% | 72.1% | 15 days | 13 | 1.5% |
| Lapsing, mid value | 25.1% | 17.9% | 162 days | 5 | 55.4% |
| New or occasional, active | 20.9% | 5.9% | 23 days | 3 | 0.0% |
| Lost, low value | 35.0% | 4.0% | 400 days | 1 | 87.6% |

**Insight:** the lapsing, mid-value cluster holds 17.9% of revenue, and the rule-based method spreads these customers across three different labels (At Risk, Loyal, Lost – High Value). Clustering groups them as one retention audience.

*Churn rate uses the Recency-based 138-day rule, so it describes the clusters rather than validating them.*

### 5. Churn modelling and validation

The primary model is a class-balanced logistic regression trained with an 80/20 stratified train/test split and `random_state=42`. It uses:

- `Frequency`
- `Monetary`

**Recency is intentionally excluded from the model features.** Since recency defines the churn label, including it would leak the answer into the model and inflate performance. The model estimates churn probability from historical purchase behaviour only.

The logistic-regression baseline recorded in the project achieved the following test-set metrics:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.6956 |
| Precision | 0.6297 |
| Recall | 0.8671 |
| F1 score | 0.7278 |

High recall is useful in a retention setting because it reduces the chance of overlooking customers who may churn. Precision should still be considered alongside outreach cost and campaign capacity.

A 300-tree, class-balanced Random Forest is also evaluated as a non-linear alternative. The implementation compares accuracy, precision, recall, F1 score, confusion matrices, and feature importance. The project retains the logistic-regression approach because it is more interpretable and, in the recorded comparison, performed better.

Finally, Pearson correlation and variance inflation factor (VIF) checks assess overlap between Frequency and Monetary. The documented VIF of 1.65 indicates acceptable multicollinearity.

### 6. Retention prioritisation

The final ranking combines likelihood and value:

`Expected revenue at risk = CLV x predicted churn probability`

Customers are sorted by this measure and the top 50 are returned with their RFM values, segment, CLV, churn probability, and expected revenue at risk. This provides a practical first audience for retention offers, account-manager calls, or personalised reactivation campaigns.

## How to run

### Prerequisites

- Python 3.10+ recommended
- A local `online_retail_II.csv` file in the project root

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn scipy cycler
```

### Execution order

The files in `Development/` are sequential analysis blocks and share in-memory dataframes. Run them in the following order in the same Jupyter/IPython session:

1. `Data Load.py`
2. `Step 1 — Data Profiling & Quality Assessment.py`
3. `Step 2 — Exploratory Data Analysis.py`
4. `rfm_analysis_segmentation.py`
5. `rfm_clv_analysis.py`
6. `churn_prediction_model.py`
7. `rfm_multicollinearity_analysis.py`
8. `random_forest_churn_model.py`
9. `retention_priority_list.py`

For example, in a notebook use IPython's `%run` command, keeping the quoted paths intact:

```python
%run "Development/Data Load.py"
%run "Development/Step 1 — Data Profiling & Quality Assessment.py"
%run "Development/Step 2 — Exploratory Data Analysis.py"
%run "Development/rfm_analysis_segmentation.py"
%run "Development/rfm_clv_analysis.py"
%run "Development/churn_prediction_model.py"
%run "Development/rfm_multicollinearity_analysis.py"
%run "Development/random_forest_churn_model.py"
%run "Development/retention_priority_list.py"
```

The analysis generates tables and Matplotlib figures in memory. Export any business-ready tables or figures after running the final block if they are needed outside the notebook session.

## Repository structure

```text
Development/
├── Data Load.py                                  # Dataset configuration and loading
├── Step 1 — Data Profiling & Quality Assessment.py
├── Step 2 — Exploratory Data Analysis.py
├── rfm_analysis_segmentation.py                  # Cleaning, RFM table, segments
├── rfm_clv_analysis.py                           # CLV calculation and segment comparison
├── churn_prediction_model.py                     # Logistic-regression churn baseline
├── rfm_multicollinearity_analysis.py             # Correlation and VIF checks
├── random_forest_churn_model.py                  # Non-linear model comparison
└── retention_priority_list.py                    # Revenue-at-risk ranking
```

## Data-science practices demonstrated

- Translating a business retention problem into measurable customer-level outcomes.
- Profiling and cleaning transactional data with documented rules.
- Feature engineering with RFM metrics, customer tenure, CLV, and churn-risk scores.
- Exploratory analysis that informs a data-based churn threshold.
- Interpretable segmentation designed for non-technical stakeholders.
- Leakage prevention by removing Recency from model inputs when it defines the target.
- Stratified validation, class balancing, multiple classification metrics, and baseline-versus-alternative model comparison.
- Multicollinearity diagnostics before interpreting linear-model coefficients.
- Converting model outputs into a ranked, revenue-aware operational decision list.

## Assumptions and limitations

- The churn label is a recency-based proxy, not a confirmed cancellation event. It should be recalibrated to each business's typical buying cycle.
- CLV is an analytical estimate, not a contractual revenue forecast. It does not include margin, acquisition cost, marketing spend, refunds beyond the cleaning rules, or discounting.
- The model is retrospective and uses only Frequency and Monetary to avoid target leakage. A production version should use time-safe snapshots and richer pre-churn features, such as category mix, channel, geography, returns, discount behaviour, and customer-service interactions.
- Before live use, choose a probability threshold based on campaign economics, validate on a later time period, monitor drift, and measure incremental retention with a holdout group.

## Next steps

1. Package the sequential blocks as a reproducible pipeline with explicit inputs and outputs.
2. Export the priority list to CSV or connect it to a dashboard/CRM workflow.
3. Add time-based validation and probability calibration.
4. Estimate net value at risk using gross margin and contact cost.
5. Test retention actions by segment to learn which offers produce incremental value.

## Data source and citation

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

The dataset is provided by UCI under the CC BY 4.0 license; please retain appropriate attribution when using or sharing derived work.
