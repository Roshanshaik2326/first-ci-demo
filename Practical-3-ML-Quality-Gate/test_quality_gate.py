
import unittest
from unittest.mock import mock_open, patch

import quality_gate


class TestQualityGate(unittest.TestCase):

    @patch("quality_gate.Path.exists", return_value=True)
    @patch("quality_gate.Path.open", new_callable=mock_open,
           read_data='{"accuracy": 0.88}')
    def test_quality_gate_passes(self, mock_file, mock_exists):
        self.assertTrue(quality_gate.check_model_quality())

    @patch("quality_gate.Path.exists", return_value=True)
    @patch("quality_gate.Path.open", new_callable=mock_open,
           read_data='{"accuracy": 0.65}')
    def test_quality_gate_fails_low_accuracy(self, mock_file, mock_exists):
        self.assertFalse(quality_gate.check_model_quality())

    @patch("quality_gate.Path.exists", return_value=True)
    @patch("quality_gate.Path.open", new_callable=mock_open,
           read_data='{"training_samples": 240}')
    def test_quality_gate_fails_missing_accuracy(self, mock_file, mock_exists):
        self.assertFalse(quality_gate.check_model_quality())

    @patch("quality_gate.Path.exists", return_value=False)
    def test_quality_gate_fails_missing_file(self, mock_exists):
        self.assertFalse(quality_gate.check_model_quality())


if __name__ == "__main__":
    unittest.main()
