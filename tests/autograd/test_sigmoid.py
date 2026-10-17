"""Sigmoid forward values, chain rule, and numerical stability."""

import unittest
import math

from aurora.autograd import AutoNode, backward


class SigmoidTests(unittest.TestCase):
    def test_forward_value_and_parent_connection(self):
        z = AutoNode(0)
        result = z.sigmoid()
        self.assertIsInstance(result, AutoNode)
        self.assertIsNot(result, z)
        self.assertEqual(result.value, 0.5)
        self.assertEqual(result.gradient, 0)
        self.assertEqual(len(result.parents), 1)
        self.assertIs(result.parents[0], z)

    def test_local_rule_scales_and_accumulates_incoming_gradient(self):
        for incoming in (0, 1, 8, -3):
            with self.subTest(incoming=incoming):
                z = AutoNode(0)
                result = z.sigmoid()
                z.gradient = 7
                result.gradient = incoming
                result.operation(result)
                self.assertEqual(z.gradient, 7 + incoming * 0.25)
                self.assertEqual((z.value, result.value), (0, 0.5))

    def test_shared_sigmoid_result_receives_all_contributions(self):
        z = AutoNode(0)
        s = z.sigmoid()
        output = s * s + s
        backward(output)
        self.assertEqual(s.gradient, 2)
        self.assertEqual(z.gradient, 0.5)

    def test_composed_gradient_matches_finite_differences(self):
        def numeric_loss(z):
            s = 0.5 * (1 + math.tanh(z / 2))
            return (s - 0.3) ** 2

        epsilon = 1e-6
        for value in (-3.0, -0.5, 0.0, 1.5, 3.0):
            with self.subTest(value=value):
                z = AutoNode(value)
                error = z.sigmoid() - AutoNode(0.3)
                loss = error * error
                backward(loss)
                expected = (
                    numeric_loss(value + epsilon) - numeric_loss(value - epsilon)
                ) / (2 * epsilon)
                self.assertAlmostEqual(z.gradient, expected, delta=1e-8)

    def test_large_finite_inputs_have_finite_outputs_and_gradients(self):
        for value, expected in ((-1000.0, 0.0), (1000.0, 1.0)):
            with self.subTest(value=value):
                z = AutoNode(value)
                result = z.sigmoid()
                self.assertTrue(math.isfinite(result.value))
                self.assertEqual(result.value, expected)
                backward(result)
                self.assertTrue(math.isfinite(z.gradient))
                self.assertAlmostEqual(z.gradient, 0.0, delta=1e-12)

