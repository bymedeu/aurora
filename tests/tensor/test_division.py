"""Verify elementwise division for equal-shaped tensors."""

import unittest

import numpy as np

from aurora.autograd import backward
from aurora.tensor import Tensor


class TensorDivisionTests(unittest.TestCase):
    def test_forward_preserves_shape_and_operand_order(self):
        cases = (
            (6.0, 2.0, 3.0),
            ([6, 0, 8], [2, -4, 4], [3, 0, 2]),
            (
                [[6, -4, 0], [8, 9, -3]],
                [[2, 2, -5], [-4, 3, -2]],
                [[3, -2, 0], [-2, 3, 1.5]],
            ),
        )
        for numerator, denominator, expected in cases:
            with self.subTest(shape=np.shape(numerator)):
                a, b = Tensor(numerator), Tensor(denominator)

                result = a / b

                self.assertEqual(result.value.shape, a.value.shape)
                np.testing.assert_array_equal(result.value, expected)
                self.assertEqual(len(result.parents), 2)
                self.assertIs(result.parents[0], a)
                self.assertIs(result.parents[1], b)
                np.testing.assert_array_equal(a.value, numerator)
                np.testing.assert_array_equal(b.value, denominator)

    def test_any_zero_denominator_raises(self):
        for denominator in (0, [2, 0], [[1, 2], [3, -0.0]]):
            with self.subTest(shape=np.shape(denominator)):
                numerator = Tensor(np.ones(np.shape(denominator)))
                with self.assertRaises(ZeroDivisionError):
                    _ = numerator / Tensor(denominator)

    def test_weighted_loss_accumulates_into_both_operand_gradients(self):
        a = Tensor([6, -8, 0], gradient=[10, 20, 30])
        b = Tensor([2, -4, 2], gradient=[-10, -20, -30])
        quotient = a / b
        loss = (quotient * Tensor([3, -2, 0.5])).sum()

        backward(loss)

        self.assertEqual(loss.value.item(), 5.0)
        np.testing.assert_array_equal(quotient.gradient, [3, -2, 0.5])
        np.testing.assert_array_equal(a.gradient, [11.5, 20.5, 30.25])
        np.testing.assert_array_equal(b.gradient, [-14.5, -21, -30])

    def test_both_operand_gradients_match_finite_differences(self):
        numerator = np.array([[6.0, -2, 0], [-4, 2, 5]])
        denominator = np.array([[1.5, 4, 2], [2, -3, -0.5]])
        weights = np.array([[2.0, -1, 3], [0.5, -2, 1]])
        a, b = Tensor(numerator), Tensor(denominator)

        backward(((a / b) * Tensor(weights)).sum())

        epsilon = 1e-6
        for operand, values in enumerate((numerator, denominator)):
            expected = np.empty_like(values)
            for index in np.ndindex(values.shape):
                offset = np.zeros_like(values)
                offset[index] = epsilon
                plus = [numerator, denominator]
                minus = [numerator, denominator]
                plus[operand] = values + offset
                minus[operand] = values - offset
                difference = np.sum(weights * plus[0] / plus[1]) - np.sum(
                    weights * minus[0] / minus[1]
                )
                expected[index] = difference / (2 * epsilon)
            gradient = (a, b)[operand].gradient
            self.assertEqual(gradient.shape, values.shape)
            np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
