# Retention Priority Dashboard: RFM, CLV, and Churn Risk

A customer retention analysis that identifies which customers are most valuable and most at risk of leaving, so retention effort and budget can be prioritized where it matters most.

## Business Problem

Retention budgets are limited, so a business needs to know not just who might churn, but who is actually worth retaining. This project combines customer value estimation with churn risk to produce a ranked, actionable retention priority list, rather than a simple churn prediction alone.

## Dataset

Online Retail II (UCI Machine Learning Repository), roughly 1 million transaction records from a UK based online retailer, covering January 2009 to December 2011.

## Approach

**1. Data Profiling and Cleaning**
Assessed data quality first: missing Customer IDs, duplicate rows, cancellations, and invalid values. Removed rows with a missing Customer ID, since customer level analysis requires a valid identity, along with duplicates and cancellation transactions. Retained about 75 percent of the original dataset after cleaning.

**2. RFM Segmentation**
Calculated Recency, Frequency, and Monetary value per customer, scored each into quartiles, and combined them into named segments such as Champions, At Risk, and Lost.

**3. Churn Definition and Modeling**
Defined churn using a data driven threshold, based on the 75th percentile of typical reorder gaps. Built a logistic regression model to predict churn probability. Identified and corrected a data leakage issue where Recency, used to define the churn label, was also included as a model feature. After removing it, the model reached 69.6 percent accuracy using Frequency and Monetary alone. Also tested a Random Forest model for comparison, which underperformed logistic regression across all metrics, indicating the underlying relationship is largely linear.

**4. Customer Lifetime Value**
Estimated CLV per customer using Average Order Value, Purchase Frequency Rate, and average customer lifespan, derived directly from the churn rate rather than an arbitrary assumption.

**5. Retention Priority Ranking**
Combined CLV and churn probability into an Expected Revenue at Risk score per customer, then ranked all customers to produce a final, prioritized retention list.

## Key Result

The top 50 customers, less than 1 percent of the total customer base, represent an estimated £9.26 million in future value, 24.7 percent of total projected customer value across all 5,878 customers. Prioritizing retention efforts on this small group protects nearly a quarter of the company's total customer value.

## Dashboard

Screenshots below show the retention priority dashboard, built in Streamlit.

*(Add screenshots here: metric cards, segment charts, churn risk scatter plot, priority list table)*

## Tech Stack

Python, pandas, scikit-learn, matplotlib, Streamlit

## How to Run Locally
