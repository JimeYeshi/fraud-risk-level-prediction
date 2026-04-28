"""Prediction utilities for credit card fraud risk-level classification."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

LOW_RISK_THRESHOLD = 20.0
HIGH_RISK_THRESHOLD = 60.0


def assign_risk_level(fraud_probability_percent: float) -> str:
    """Convert fraud probability percentage into a business-friendly risk level."""
    if fraud_probability_percent < LOW_RISK_THRESHOLD:
        return "Low"
    if fraud_probability_percent < HIGH_RISK_THRESHOLD:
        return "Medium"
    return "High"


def load_model(model_path: str | Path):
    """Load a saved sklearn model pipeline."""
    return joblib.load(model_path)


def predict_risk(model, input_data: pd.DataFrame) -> pd.DataFrame:
    """Return fraud probability as percentage and Low/Medium/High risk level."""
    probabilities = model.predict_proba(input_data)[:, 1]
    output = input_data.copy()
    output["fraud_probability_percent"] = (probabilities * 100).round(2)
    output["legitimate_probability_percent"] = ((1 - probabilities) * 100).round(2)
    output["fraud_risk_level"] = output["fraud_probability_percent"].apply(assign_risk_level)
    return output


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "fraud_risk_model.pkl"
    data_path = project_root / "data" / "credit_card_fraud_10k.csv"
    output_path = project_root / "reports" / "sample_predictions.csv"

    model = load_model(model_path)
    df = pd.read_csv(data_path)
    features = df.drop(columns=["is_fraud", "transaction_id"], errors="ignore")
    predictions = predict_risk(model, features.head(20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(predictions[["fraud_probability_percent", "legitimate_probability_percent", "fraud_risk_level"]].head())
