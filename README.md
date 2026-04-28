# Credit Card Fraud Risk-Level Prediction

This project analyzes a credit card transaction dataset and trains a machine learning model to predict fraud risk. The final output includes fraud probability in percentage form and a business-friendly risk level: **Low**, **Medium**, or **High**.

## Project Objective

The goal is to help identify transactions that may require fraud review by using transaction characteristics such as amount, transaction hour, merchant category, foreign transaction status, location mismatch, device trust score, velocity in the last 24 hours, and cardholder age.

## Dataset

- File: `data/credit_card_fraud_10k.csv`
- Rows: 10,000 transactions
- Target column: `is_fraud`
- Class imbalance: fraud cases are much fewer than non-fraud cases, so the model uses class balancing.

## Risk-Level Logic

The model predicts the probability that a transaction is fraudulent. The probability is converted into percentages and assigned to a risk level:

| Fraud Probability | Risk Level |
|---:|---|
| Less than 20% | Low |
| 20% to less than 60% | Medium |
| 60% and above | High |



<img width="707" height="473" alt="9e8c81d1-459c-4903-b0ee-02f9348ca831" src="https://github.com/user-attachments/assets/e80ab291-3828-44bb-9dfd-6a4707ebd08d" />


## Model Used

The project uses a `RandomForestClassifier` with class balancing. This is suitable for fraud detection because the target variable is highly imbalanced.
<img width="485" height="334" alt="image" src="https://github.com/user-attachments/assets/bb162de2-79f7-4c9d-88db-3df3b9871cb6" />


## Key Outputs

- Complete exploratory data analysis
- Fraud and non-fraud distribution
- Feature analysis
- Trained machine learning model
- Model evaluation using ROC-AUC, Average Precision, classification report, and confusion matrix
- Fraud probability percentages
- Low/Medium/High fraud risk levels
