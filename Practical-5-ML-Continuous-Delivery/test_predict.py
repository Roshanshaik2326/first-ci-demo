
import unittest
from unittest.mock import patch

import pandas as pd

from predict import predict_result


class FakeModel:
    def predict(self, data):
        return [1]


class TestPredictionApplication(unittest.TestCase):

    @patch("predict.MODEL_PATH.exists", return_value=True)
    @patch("predict.joblib.load", return_value=FakeModel())
    def test_predict_pass(self, mock_load, mock_exists):
        result = predict_result(0.9, 0.8, 0.7, 0.85)
        self.assertEqual(result, "Predicted result: PASS")

    @patch("predict.MODEL_PATH.exists", return_value=True)
    @patch("predict.joblib.load")
    def test_predict_fail(self, mock_load, mock_exists):
        class FailModel:
            def predict(self, data):
                return [0]

        mock_load.return_value = FailModel()

        result = predict_result(0.4, 0.3, 0.5, 0.2)
        self.assertEqual(result, "Predicted result: FAIL")

    @patch("predict.MODEL_PATH.exists", return_value=False)
    def test_missing_model(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            predict_result(0.9, 0.8, 0.7, 0.85)


if __name__ == "__main__":
    unittest.main()
