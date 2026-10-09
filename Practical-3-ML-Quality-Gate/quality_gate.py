
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR.parent / "Practical-2-ML-CI" / "metrics.json"

MINIMUM_ACCURACY = 0.80


def check_model_quality():
    if not METRICS_PATH.exists():
        print(f"ERROR: Metrics file not found: {METRICS_PATH}")
        return False

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    accuracy = metrics.get("accuracy")

    if accuracy is None:
        print("ERROR: Accuracy metric is missing.")
        return False

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Minimum required accuracy: {MINIMUM_ACCURACY:.4f}")

    if accuracy >= MINIMUM_ACCURACY:
        print("QUALITY GATE PASSED")
        return True

    print("QUALITY GATE FAILED")
    return False


if __name__ == "__main__":
    sys.exit(0 if check_model_quality() else 1)
