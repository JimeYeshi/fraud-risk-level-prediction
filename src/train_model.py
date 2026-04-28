"""Train a machine learning model for credit card fraud-risk prediction."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predict_fraud_risk import predict_risk


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "credit_card_fraud_10k.csv"
    model_path = project_root / "models" / "fraud_risk_model.pkl"
    prediction_path = project_root / "reports" / "fraud_risk_predictions.csv"
    metrics_path = project_root / "reports" / "model_metrics.txt"

    df = pd.read_csv(data_path)
    X = df.drop(columns=["is_fraud", "transaction_id"], errors="ignore")
    y = df["is_fraud"]

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    predictions = predict_risk(pipeline, X_test)
    predictions["actual_is_fraud"] = y_test.values
    predictions.to_csv(prediction_path, index=False)

    metrics = []
    metrics.append("Credit Card Fraud Risk Model Metrics")
    metrics.append("=" * 40)
    metrics.append(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    metrics.append(f"Average Precision: {average_precision_score(y_test, y_prob):.4f}")
    metrics.append("\nConfusion Matrix:")
    metrics.append(str(confusion_matrix(y_test, y_pred)))
    metrics.append("\nClassification Report:")
    metrics.append(classification_report(y_test, y_pred, digits=4))
    metrics_path.write_text("\n".join(metrics), encoding="utf-8")

    print("Model saved to:", model_path)
    print("Predictions saved to:", prediction_path)
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()
