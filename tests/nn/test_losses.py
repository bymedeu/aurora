"""Check scalar MSE values, averaging, and differentiability."""

import unittest

from aurora.autograd import AutoNode, backward


class MeanSquaredErrorTests(unittest.TestCase):
    def setUp(self):
        from aurora.nn.losses import mean_squared_error

        self.mse = mean_squared_error

    def test_forward_value_and_prediction_gradients(self):
        predictions = [AutoNode(3), AutoNode(7)]
        targets = [AutoNode(5), AutoNode(6)]

        loss = self.mse(predictions, targets)

        self.assertIsInstance(loss, AutoNode)
        self.assertEqual(loss.value, 2.5)
        self.assertEqual([p.gradient for p in predictions], [0, 0])
        backward(loss)
        self.assertEqual([p.gradient for p in predictions], [-2, 1])

    def test_perfect_predictions_have_zero_loss_and_gradients(self):
        predictions = [AutoNode(-2), AutoNode(3)]

        loss = self.mse(predictions, [AutoNode(-2), AutoNode(3)])
        backward(loss)

        self.assertEqual(loss.value, 0)
        self.assertEqual([p.gradient for p in predictions], [0, 0])

    def test_duplicating_a_pair_preserves_mean_and_shared_gradient(self):
        prediction = AutoNode(3)
        target = AutoNode(5)

        loss = self.mse([prediction, prediction], [target, target])
        backward(loss)

        self.assertEqual(loss.value, 4)
        self.assertEqual(prediction.gradient, -4)

    def test_prediction_gradients_match_finite_differences(self):
        values = [-1.0, 0.5, 2.0]
        targets = [0.5, -0.5, 1.0]
        predictions = [AutoNode(value) for value in values]
        loss = self.mse(predictions, [AutoNode(value) for value in targets])
        backward(loss)

        def numeric_loss(predictions):
            return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / 3

        epsilon = 1e-6
        for index, prediction in enumerate(predictions):
            with self.subTest(index=index):
                above, below = list(values), list(values)
                above[index] += epsilon
                below[index] -= epsilon
                expected = (numeric_loss(above) - numeric_loss(below)) / (2 * epsilon)
                self.assertAlmostEqual(prediction.gradient, expected, delta=1e-8)

    def test_public_package_export(self):
        from aurora.nn import mean_squared_error

        self.assertIs(mean_squared_error, self.mse)
