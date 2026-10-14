"""Verify that scalar MSE retains connections to model parameters."""

import unittest

from aurora.autograd import AutoNode, backward


class MeanSquaredErrorTests(unittest.TestCase):
    def test_loss_value_and_gradients_reach_model_parameters(self):
        from aurora.nn.losses import mean_squared_error

        weight, bias = AutoNode(2), AutoNode(1)
        predictions = [weight * AutoNode(x) + bias for x in (1, 3)]
        targets = [AutoNode(5), AutoNode(6)]

        loss = mean_squared_error(predictions, targets)

        self.assertIsInstance(loss, AutoNode)
        self.assertEqual(loss.value, 2.5)
        self.assertEqual((weight.gradient, bias.gradient), (0, 0))

        backward(loss)

        self.assertEqual([p.gradient for p in predictions], [-2, 1])
        self.assertEqual(weight.gradient, 1)
        self.assertEqual(bias.gradient, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
