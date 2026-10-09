
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_result_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"


def create_dataset():
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    features = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score",
    ]

    return pd.DataFrame(X, columns=features), pd.Series(y)


def train_model():
    X, y = create_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42)),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "accuracy": float(accuracy),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=4))

    print("Model training completed.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    return model, metrics


if __name__ == "__main__":
    train_model()
