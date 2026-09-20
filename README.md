# credit-card-fraud-detection
Machine learning project to detect fraudulent credit card transactions on a highly imbalanced dataset.

## Problem
Credit card fraud is rare but costly: in this dataset, only **0.17%** of transactions (492 out of 284,807) are fraudulent. A naive model that predicts "no fraud" for everything would be 99.8% accurate — and completely useless. The real challenge is correctly identifying the rare fraud cases without flagging too many legitimate transactions.

## Dataset
- [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions, 31 columns (`V1`–`V28` are PCA-anonymized features, plus `Time`, `Amount`, `Class`)

## Approach
1. **EDA** — examined class imbalance, transaction amount and time patterns, and correlations with fraud
2. **Train/test split** — 80/20 split with stratification to preserve class ratio in both sets
3. **SMOTE** — applied oversampling on the training set only (never on test data, to avoid data leakage and keep evaluation realistic)
4. **Models** — trained and compared Logistic Regression (baseline) and Random Forest
5. **Evaluation** — used Precision, Recall, F1, ROC-AUC, and Confusion Matrix instead of accuracy, since accuracy is misleading on imbalanced data
6. **Business impact** — translated the confusion matrix into an estimated dollar impact

## Results (Random Forest)
| Metric | Score |
|---|---|
| Precision (fraud) | 83% |
| Recall (fraud) | 83% |
| ROC-AUC | 0.96 |

**Business impact:**
- Correctly caught fraud worth **$6,742**
- Missed fraud worth **$3,903**
- False alarms on legitimate transactions: 17 (**$1,604**)
- Estimated **net benefit: ~$6,657**

## Tech stack
Python · Pandas · Scikit-learn · imbalanced-learn (SMOTE) · Matplotlib · Seaborn

## Notebook
See [`credit-card-fraud-detection.ipynb`](./credit-card-fraud-detection.ipynb) for the full analysis.
