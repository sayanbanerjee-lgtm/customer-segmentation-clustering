# Customer Segmentation Using RFM Analysis, PCA and Clustering

<escape><div align="center"></escape>

[![Live Dashboard](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://customer-segmentation-clustering-rfm.streamlit.app/)

**[🚀 Launch the Live Dashboard](https://customer-segmentation-clustering-rfm.streamlit.app/)**

<escape></div></escape>

## 1. Project Overview

This project focuses on segmenting customers of an online retail business based on their purchasing behaviour using statistical analysis and machine learning techniques.

The project uses the **UCI Online Retail II dataset**, which contains transaction records from a UK-based online retailer between December 2009 and December 2011.

The primary objective is to identify meaningful customer groups using Recency, Frequency and Monetary (RFM) analysis, Principal Component Analysis (PCA), and clustering algorithms. The resulting customer segments are interpreted from a business perspective to support targeted marketing strategies and customer relationship management.

An interactive dashboard developed using Streamlit enables users to explore customer segments, analyse purchasing behaviour, look up customers and predict the segment of a new customer.

## 2. Dataset Information

**Dataset:** UCI Online Retail II

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

**Original dataset size:** 1,067,371 transaction records

**Transaction period:** 1 December 2009 – 9 December 2011

**Country:** United Kingdom-based online retailer

The dataset contains transaction-level information, including invoice numbers, product codes, product descriptions, quantities, invoice dates, unit prices, customer IDs and countries.

### Dataset citation

Chen, D. (2012). _Online Retail II_. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

## 3. Data Cleaning and Preparation

The raw dataset was examined for missing values, duplicate records, cancelled invoices, negative quantities, zero quantities and invalid unit prices.

The following cleaning decisions were applied:

- Removed duplicate transaction records.
- Excluded transactions without Customer IDs.
- Excluded cancelled invoices.
- Excluded records with non-positive quantities.
- Excluded records with non-positive unit prices.
- Retained valid positive-sales transactions.

After cleaning:

| Metric                       |         Result |
| ---------------------------- | -------------: |
| Original transaction records |      1,067,371 |
| Valid purchase records       |        779,425 |
| Unique customers             |          5,878 |
| Unique invoices              |         36,969 |
| Total positive sales value   | £17,374,804.27 |

The original raw data was preserved separately from the cleaned dataset.

## 4. RFM Analysis

Customer-level RFM metrics were calculated from the cleaned transaction data.

- **Recency:** Number of days since the customer's most recent purchase.
- **Frequency:** Number of distinct invoices associated with the customer.
- **Monetary:** Total purchase value calculated as quantity multiplied by unit price.

The reference date was set to 10 December 2011, one day after the latest transaction date.

The resulting RFM dataset contained 5,878 customers and three behavioural variables.

## 5. Data Transformation and Standardization

Distributional analysis indicated substantial positive skewness in Frequency and Monetary.

Box-Cox transformations were therefore applied to Frequency and Monetary, while Recency was retained without transformation.

The transformed RFM variables were standardized using StandardScaler to ensure comparable scales during dimensionality reduction and clustering.

## 6. Principal Component Analysis (PCA)

PCA was performed on the standardized RFM variables to reduce dimensionality while retaining most of the information.

The first two principal components explained:

| Component  | Explained Variance |
| ---------- | -----------------: |
| PC1        |             75.16% |
| PC2        |             19.57% |
| Cumulative |             94.72% |

The first two principal components were therefore retained for the primary K-Means clustering analysis.

## 7. Customer Clustering

### K-Means Clustering

K-Means clustering was evaluated for cluster counts ranging from 2 to 10 using the elbow method and silhouette scores.

Solutions with 2, 3 and 5 clusters were examined further for business interpretability.

The five-cluster solution was selected as the final segmentation because it provided useful behavioural distinctions while maintaining reasonably balanced cluster sizes.

### Hierarchical Clustering

Agglomerative hierarchical clustering was also investigated using different distance metrics and linkage methods.

A representative sample of 1,000 customers was used for hierarchical analysis due to its computational complexity.

For a comparable five-cluster evaluation on the same sample:

- K-Means silhouette score: 0.3761
- Ward hierarchical clustering silhouette score: 0.3032
- Adjusted Rand Index: 0.4519
- Normalized Mutual Information: 0.5997

K-Means was selected as the final model for customer segmentation and new-customer prediction.

## 8. Final Customer Segments

The five-cluster K-Means solution identified the following customer groups:

| Cluster | Segment Name                              | Customer Share | Revenue Contribution |
| ------- | ----------------------------------------- | -------------: | -------------------: |
| 0       | High-Value Loyal Customers                |         23.17% |               78.41% |
| 1       | Recent One-Time / Low-Frequency Customers |         16.52% |                1.76% |
| 2       | Regular Moderate-Value Customers          |         28.17% |               12.03% |
| 3       | Dormant Repeat Customers                  |         14.68% |                6.27% |
| 4       | Long-Inactive Low-Value Customers         |         17.45% |                1.52% |

These segments reveal differences in customer engagement, purchasing frequency, spending behaviour and inactivity.

## 9. Business Recommendations

The identified customer groups suggest different marketing and retention strategies:

- **High-Value Loyal Customers:** VIP loyalty programmes, exclusive offers and personalized recommendations.
- **Recent One-Time / Low-Frequency Customers:** Second-purchase incentives, welcome-back campaigns and targeted follow-ups.
- **Regular Moderate-Value Customers:** Loyalty rewards, cross-selling, upselling and product bundles.
- **Dormant Repeat Customers:** Win-back campaigns and personalized reactivation offers.
- **Long-Inactive Low-Value Customers:** Low-cost reactivation campaigns and limited promotional incentives.

## 10. Interactive Streamlit Dashboard

The project includes an interactive dashboard developed using Streamlit and Plotly.

### Dashboard features

- Project overview and dataset information.
- Customer and transaction summary metrics.
- RFM distribution visualizations.
- Customer segment analysis and comparisons.
- PCA-based customer segmentation visualization.
- RFM radar chart.
- Customer lookup by Customer ID.
- New-customer segment prediction.
- Purchase interval analysis.
- Segment-specific business recommendations.
- Cleaned dataset preview and download.

The new-customer prediction feature reuses the fitted Box-Cox transformation parameters, scaler, PCA model and K-Means model to ensure consistent predictions.

## 11. Technologies and Libraries

- Python
- Pandas and NumPy
- SciPy
- Matplotlib and Seaborn
- Scikit-learn
- Plotly
- Streamlit
- Jupyter Notebook
- Joblib

## 12. Project Structure

```text
IDEAS_TIH_CustomerSegmentation/
│
├── README.md
|
│── app.py
│
├── cleaned_purchases.csv
├── customer_rfm.csv
|── customer_pca_scores.csv
│
│── customer_segmentation_model.joblib
|
│── dataCleaning.ipynb
│
└── requirements.txt
```

## 13. Installation and Execution

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd IDEAS_TIH_CustomerSegmentation
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Launch the Streamlit dashboard

```bash
streamlit run app/app.py
```

The dashboard will open in your default web browser.

**Note:** The application requires the processed dataset and trained model artifacts to be present in their expected project directories.

## 14. Conclusion

This project demonstrates how statistical techniques and machine learning algorithms can be combined to understand customer purchasing behaviour and develop actionable customer segments.

The analysis identifies distinct customer groups with different engagement and spending patterns. The resulting segmentation provides a foundation for targeted marketing, customer retention and data-driven business decision-making.
