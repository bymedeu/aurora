"""Show the boundary between loss construction, backward, and parameter update."""

import unittest

from aurora.autograd import AutoNode, backward


class LearningStepTests(unittest.TestCase):
    def test_existing_components_reduce_loss_without_linking_update_graphs(self):
        from aurora.nn.losses import mean_squared_error
        from aurora.optim.gradient_descent import gradient_descent_step

        weight, bias = AutoNode(2), AutoNode(1)
        inputs = [AutoNode(1), AutoNode(3)]
        targets = [AutoNode(5), AutoNode(6)]
        predictions = [weight * x + bias for x in inputs]
        loss = mean_squared_error(predictions, targets)
        self.assertEqual(loss.value, 2.5)
        self.assertEqual((weight.gradient, bias.gradient), (0, 0))

        backward(loss)
        self.assertEqual((weight.gradient, bias.gradient), (1, -1))
        self.assertEqual((weight.value, bias.value), (2, 1))

        new_weight, new_bias = gradient_descent_step([weight, bias], 0.1)
        self.assertEqual((new_weight.parents, new_bias.parents), ((), ()))
        self.assertEqual((new_weight.gradient, new_bias.gradient), (0, 0))
        next_predictions = [new_weight * x + new_bias for x in inputs]
        next_loss = mean_squared_error(next_predictions, targets)
        self.assertAlmostEqual(next_loss.value, 2.32)
        self.assertLess(next_loss.value, loss.value)

        backward(next_loss)
        self.assertEqual((weight.gradient, bias.gradient), (1, -1))
        self.assertAlmostEqual(new_weight.gradient, 0.4)
        self.assertAlmostEqual(new_bias.gradient, -1.2)
