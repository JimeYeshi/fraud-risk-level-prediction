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

#Complete exploratory data analysis

<img width="707" height="473" alt="d424e102-f2b7-4124-9ec1-a5014865629d" src="https://github.com/user-attachments/assets/eb3ef94a-36cd-4e33-a5ae-b16560f659e1" />
<img width="699" height="473" alt="0e49501b-f1dc-4456-9421-e9d8ac72cad4" src="https://github.com/user-attachments/assets/7e00279f-c28c-44ab-8ccb-d64c5e8ad1c8" />
<img width="707" height="473" alt="9a49f797-e782-48ff-87ac-3534a1b12956" src="https://github.com/user-attachments/assets/5a2c421f-5cc6-4e45-9804-59974f02dada" />
<img width="699" height="473" alt="0c602f55-c5d6-4d8d-b4fc-22f6eac760cf" src="https://github.com/user-attachments/assets/51f8287a-1d54-497f-b9f9-87f26a145919" />
<img width="699" height="473" alt="598f8c1c-7956-40ff-8461-a76985f3e7b7" src="https://github.com/user-attachments/assets/7b5a764d-3ec7-41b2-b8c7-4d5e8feea951" />
<img width="703" height="522" alt="2da434dd-922b-40c2-a694-003c8d1311e0" src="https://github.com/user-attachments/assets/0c73f5b9-c33c-43e5-8bfe-06df466d39e2" />

# Fraud and non-fraud distribution

  <img width="716" height="473" alt="646e4490-d0c6-4002-bdc8-f709915b02cd" src="https://github.com/user-attachments/assets/e18cc914-a670-4d77-af18-80da5c034c36" />

# Feature analysis
  
<img width="743" height="568" alt="a7abe71c-607c-47c2-82cf-03fcce7d2471" src="https://github.com/user-attachments/assets/c70b40f7-5bd8-4204-a81f-d8b861c3b6d1" />

- Trained machine learning model
# Model evaluation using ROC-AUC, Average Precision, classification report, and confusion matrix
<img width="661" height="473" alt="a7af1352-290e-41da-8fa4-81dbb5232186" src="https://github.com/user-attachments/assets/314068b1-dbd3-47d2-b906-e08d50723492" />
<img width="695" height="473" alt="0f0c45d9-24de-4d35-be53-ff9621e9fc27" src="https://github.com/user-attachments/assets/fd99d72e-c0b2-4797-89a5-1057d967f0ea" />

- Fraud probability percentages
- Low/Medium/High fraud risk levels
