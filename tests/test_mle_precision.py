import unittest

import numpy as np

from netcal.scaling import LogisticCalibration, TemperatureScaling


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(selected, 1e-15, 1.0))))


class TestMLEPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        num_samples, num_classes = 2000, 5
        latent_logits = rng.normal(size=(num_samples, num_classes))
        true_probabilities = softmax(latent_logits)
        cls.labels = np.array([
            rng.choice(num_classes, p=probabilities)
            for probabilities in true_probabilities
        ])

        # Make the predictions deliberately overconfident. Both scaling methods
        # should move away from their identity initialization and lower NLL.
        cls.predictions = softmax(3.0 * latent_logits)

    def test_mle_optimizes_float32_and_float64_inputs(self):
        for calibrator_type in (TemperatureScaling, LogisticCalibration):
            losses = {}
            for dtype in (np.float32, np.float64):
                with self.subTest(calibrator=calibrator_type.__name__, dtype=dtype.__name__):
                    predictions = self.predictions.astype(dtype)
                    before = negative_log_likelihood(predictions, self.labels)

                    calibrator = calibrator_type(method="mle", use_cuda=False)
                    calibrator.fit(predictions, self.labels, random_state=42, tensorboard=False)
                    calibrated = calibrator.transform(predictions)
                    after = negative_log_likelihood(calibrated, self.labels)
                    losses[dtype] = after

                    self.assertLess(after, before - 0.2)
                    self.assertGreater(np.max(np.abs(calibrated - predictions)), 1e-3)

            self.assertAlmostEqual(losses[np.float32], losses[np.float64], places=4)


if __name__ == "__main__":
    unittest.main()
