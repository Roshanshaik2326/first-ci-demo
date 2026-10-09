
import unittest

from train_model import create_dataset, train_model


class TestMLPipeline(unittest.TestCase):

    def test_dataset_shape(self):
        X, y = create_dataset()

        self.assertEqual(X.shape, (300, 4))
        self.assertEqual(len(y), 300)

    def test_dataset_has_expected_features(self):
        X, _ = create_dataset()

        expected_features = [
            "attendance",
            "internal_marks",
            "assignment_marks",
            "previous_score",
        ]

        self.assertEqual(list(X.columns), expected_features)

    def test_model_training(self):
        model, metrics = train_model()

        self.assertIsNotNone(model)
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
