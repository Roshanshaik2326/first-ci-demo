
import argparse
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR.parent
    / "Practical-2-ML-CI"
    / "student_result_model.pkl"
)

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]


def predict_result(attendance, internal_marks, assignment_marks, previous_score):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run train_model.py in Practical-2-ML-CI first."
        )

    model = joblib.load(MODEL_PATH)

    student_data = pd.DataFrame(
        [[attendance, internal_marks, assignment_marks, previous_score]],
        columns=FEATURES,
    )

    prediction = int(model.predict(student_data)[0])

    if prediction == 1:
        return "Predicted result: PASS"

    return "Predicted result: FAIL"


def main():
    parser = argparse.ArgumentParser(
        description="Predict a student result using the trained ML model."
    )

    parser.add_argument("--attendance", type=float, required=True)
    parser.add_argument("--internal-marks", type=float, required=True)
    parser.add_argument("--assignment-marks", type=float, required=True)
    parser.add_argument("--previous-score", type=float, required=True)

    args = parser.parse_args()

    result = predict_result(
        args.attendance,
        args.internal_marks,
        args.assignment_marks,
        args.previous_score,
    )

    print(result)


if __name__ == "__main__":
    main()
