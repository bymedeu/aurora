"""Check integer powers against their values and analytical derivatives."""

import unittest

from aurora.autograd import AutoNode, backward


class PowerTests(unittest.TestCase):
    def test_positive_integer_values_and_gradients(self):
        for exponent in (1, 2, 3, 4, 15, 16, 31):
            for value in (-2.0, -0.5, 0.0, 0.5, 2.0):
                with self.subTest(exponent=exponent, value=value):
                    base = AutoNode(value)
                    result = base ** exponent
                    self.assertIsInstance(result, AutoNode)
                    self.assertEqual(result.value, value ** exponent)
                    self.assertEqual(base.gradient, 0)
                    self.assertEqual(result.gradient, 0)

                    backward(result)

                    self.assertEqual(base.gradient, exponent * value ** (exponent - 1))
                    self.assertEqual(base.value, value)

    def test_zero_exponent_is_constant_one(self):
        # The implemented zero-exponent extension follows Python's x ** 0 = 1.
        for value in (-2.0, 0.0, 2.0):
            with self.subTest(value=value):
                base = AutoNode(value)
                base.gradient = 7
                result = base ** 0
                self.assertEqual(result.value, 1)

                backward(result)

                self.assertEqual(base.gradient, 7)

    def test_incoming_gradient_is_scaled_and_accumulated(self):
        for exponent in (2, 3, 15):
            for incoming in (0, 1, 4, -3):
                with self.subTest(exponent=exponent, incoming=incoming):
                    base = AutoNode(-1.5)
                    base.gradient = 7
                    output = (base ** exponent) * AutoNode(incoming)

                    backward(output)

                    expected = 7 + incoming * exponent * (-1.5) ** (exponent - 1)
                    self.assertAlmostEqual(base.gradient, expected)

    def test_shared_power_receives_all_contributions(self):
        base = AutoNode(2)
        shared = base ** 3
        output = shared + shared * base

        backward(output)

        # f(x) = x^3 + x^4; f'(2) = 3*2^2 + 4*2^3.
        self.assertEqual(shared.gradient, 3)
        self.assertEqual(base.gradient, 44)

    def test_different_exponents_share_the_same_base(self):
        base = AutoNode(2)
        output = base ** 2 + base ** 3 + base ** 15

        backward(output)

        self.assertEqual(output.value, 4 + 8 + 32768)
        self.assertEqual(base.gradient, 4 + 12 + 15 * 2 ** 14)

    def test_composed_parameter_gradients_match_finite_differences(self):
        def expression(a, b):
            return (a * b + a) ** 3 + (a - b) ** 2

        epsilon = 1e-6
        for av, bv in ((0.5, 1.2), (-0.7, 0.4), (0.0, 2.0)):
            with self.subTest(a=av, b=bv):
                a, b = AutoNode(av), AutoNode(bv)

                backward(expression(a, b))

                expected_a = (
                    expression(av + epsilon, bv) - expression(av - epsilon, bv)
                ) / (2 * epsilon)
                expected_b = (
                    expression(av, bv + epsilon) - expression(av, bv - epsilon)
                ) / (2 * epsilon)
                self.assertAlmostEqual(a.gradient, expected_a, delta=1e-6)
                self.assertAlmostEqual(b.gradient, expected_b, delta=1e-6)

    def test_negative_integer_exponents_are_rejected(self):
        for exponent in (-1, -3):
            with self.subTest(exponent=exponent):
                base = AutoNode(2)
                with self.assertRaises(ValueError):
                    base ** exponent
                self.assertEqual(base.gradient, 0)

    def test_non_integer_exponents_are_rejected(self):
        for exponent in (1.5, 2.0, -0.5, 0.0, "2", None, AutoNode(2)):
            with self.subTest(exponent=exponent):
                base = AutoNode(2)
                with self.assertRaises(TypeError):
                    base ** exponent
                self.assertEqual(base.gradient, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
