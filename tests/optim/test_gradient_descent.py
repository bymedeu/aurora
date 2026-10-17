"""Check numerical updates independently of gradient computation."""

import unittest

from aurora.autograd import AutoNode


class GradientDescentTests(unittest.TestCase):
    def setUp(self):
        from aurora.optim.gradient_descent import gradient_descent_step

        self.step = gradient_descent_step

    def test_updates_follow_gradient_sign_and_learning_rate(self):
        parameters = [AutoNode(2), AutoNode(-1), AutoNode(0.5)]
        for parameter, gradient in zip(parameters, (4, 0, -2)):
            parameter.gradient = gradient

        updated = self.step(parameters, 0.25)

        self.assertEqual([p.value for p in updated], [1, -1, 1])

    def test_returns_fresh_zero_gradient_leaves(self):
        parameters = [AutoNode(2), AutoNode(1)]
        parameters[0].gradient, parameters[1].gradient = 1, -1

        updated = self.step(parameters, 0.1)

        self.assertIsNot(updated, parameters)
        self.assertAlmostEqual(updated[0].value, 1.9)
        self.assertAlmostEqual(updated[1].value, 1.1)
        for old, new in zip(parameters, updated):
            self.assertIsNot(new, old)
            self.assertEqual(new.parents, ())
            self.assertEqual(new.gradient, 0)

    def test_preserves_original_parameter_values_gradients_and_graph(self):
        parameter = AutoNode(3)
        parameter.gradient = 5
        parents, operation = parameter.parents, parameter.operation

        self.step([parameter], 0.1)

        self.assertEqual(parameter.value, 3)
        self.assertEqual(parameter.gradient, 5)
        self.assertIs(parameter.parents, parents)
        self.assertIs(parameter.operation, operation)

    def test_zero_gradient_keeps_the_numerical_value(self):
        parameter = AutoNode(7)

        updated = self.step([parameter], 0.1)

        self.assertEqual(updated[0].value, 7)
        self.assertEqual(updated[0].gradient, 0)

    def test_public_package_export(self):
        from aurora.optim import gradient_descent_step

        self.assertIs(gradient_descent_step, self.step)
