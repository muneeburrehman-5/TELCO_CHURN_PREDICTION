# TELCO_CHURN_PREDICTION

A complete machine learning classification project to predict telecom customer churn using the IBM Telco Customer Churn dataset.

## 1) Task Objective

The goal of this project is to predict whether a telecom customer is likely to churn (`Yes/No`) using account, service, and billing information.

This is a **binary classification** problem with **class imbalance** (roughly 73% non-churn vs 27% churn), so the objective is not just high accuracy, but strong performance on the churn class through metrics such as:

- Precision
- Recall
- F1-score
- ROC-AUC

In addition to prediction, the project also aims to identify key churn-driving factors that can support retention strategy and business decision-making.

---

## 2) My Approach

### A. Data Understanding and Exploration
- Loaded and inspected the IBM Telco churn dataset (`7043` rows, `21` columns).
- Reviewed schema, data types, summary statistics, and class distribution.
- Confirmed target imbalance (`Churn`: ~26.5% positive class).

### B. Data Preprocessing
- Converted `TotalCharges` from string to numeric.
- Detected and handled `11` blank `TotalCharges` entries (customers with `tenure = 0`) by filling with `0`.
- Dropped `customerID` (non-predictive identifier).
- Standardized categorical values like:
  - `No internet service` → `No`
  - `No phone service` → `No`
- Encoded features:
  - Binary categorical columns via label encoding.
  - Multi-category columns (`InternetService`, `Contract`, `PaymentMethod`) via one-hot encoding (`drop_first=True`).
- Prepared train/test splits and scaling pipeline for algorithms that require feature scaling.

### C. Modeling Strategy
Multiple supervised classification models were trained and compared, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)

Evaluation emphasized churn-class performance rather than relying only on accuracy.

### D. Evaluation
Used a broad metric set for robust comparison:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report
- ROC-AUC

This ensured reliable performance assessment under class imbalance.

---

## 3) Results and Insights

### Model Performance Insights
- Accuracy alone was not treated as the final criterion due to class imbalance.
- Models were compared based on their ability to correctly identify churners (recall) while maintaining precision.
- Ensemble models (especially tree-based approaches like Random Forest / Gradient Boosting) generally provide stronger non-linear learning and often perform competitively for churn problems.

### Business Insights from the Workflow
From preprocessing and exploratory patterns in this dataset, churn risk is typically associated with:

- **Month-to-month contracts** (higher churn tendency)
- **Electronic check payment method**
- **Lower tenure** (newer customers are more likely to churn)
- Certain service/billing combinations that increase dissatisfaction risk

### Practical Retention Recommendations
A retention team can act by:

- Targeting early-life customers with onboarding and engagement offers
- Incentivizing migration from month-to-month to longer contracts
- Designing intervention campaigns for high-risk payment/service segments
- Monitoring high-risk cohorts using model scoring pipelines

---

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **ML Tasks:** preprocessing, feature encoding, scaling, classification, model comparison, evaluation

---

## Dataset

- **Source:** IBM Sample Data Sets
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Target Variable:** `Churn`

---

## Repository Structure (expected)

- `TELCO_CHURN_PREDICTION_.ipynb` — end-to-end notebook
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — dataset
- `README.md` — project documentation

---

## Conclusion

This project demonstrates a complete churn-prediction workflow: from raw data handling to model benchmarking and business-oriented interpretation.  
It is a strong practical example of applying machine learning to customer retention analytics in real-world subscription businesses.
