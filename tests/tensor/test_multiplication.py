"""Equal-shape products, accumulated derivatives, and graph integration."""

import unittest

import numpy as np

from aurora.autograd import backward
from aurora.tensor import Tensor


class TensorMultiplicationTests(unittest.TestCase):
    def test_forward_preserves_shape_and_parent_identity(self):
        cases = (
            (2.5, -2.0, -5.0),
            ([1, -2, 3], [4, 5, -6], [4, -10, -18]),
            ([[1, 2], [3, 4]], [[5, 6], [7, 8]], [[5, 12], [21, 32]]),
            (np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))),
        )
        for left, right, expected in cases:
            with self.subTest(shape=np.shape(left)):
                a, b = Tensor(left), Tensor(right)
                result = a * b
                self.assertIsInstance(result, Tensor)
                self.assertEqual(result.value.shape, a.value.shape)
                self.assertEqual(result.value.dtype, np.dtype(np.float64))
                np.testing.assert_array_equal(result.value, expected)
                self.assertEqual(len(result.parents), 2)
                self.assertIs(result.parents[0], a)
                self.assertIs(result.parents[1], b)
                self.assertEqual(result.gradient.shape, result.value.shape)
                np.testing.assert_array_equal(
                    result.gradient, np.zeros(result.value.shape)
                )

    def test_forward_preserves_input_values_and_existing_gradients(self):
        a = Tensor([2, 3], gradient=[7, -3])
        b = Tensor([4, 5], gradient=[2, 9])
        result = a * b
        np.testing.assert_array_equal(a.value, [2, 3])
        np.testing.assert_array_equal(b.value, [4, 5])
        np.testing.assert_array_equal(a.gradient, [7, -3])
        np.testing.assert_array_equal(b.gradient, [2, 9])
        self.assertFalse(np.shares_memory(result.value, a.value))
        self.assertFalse(np.shares_memory(result.value, b.value))

    def test_local_rule_accumulates_nonuniform_received_gradients(self):
        a = Tensor([[2, 3], [4, 5]], gradient=[[1, 2], [3, 4]])
        b = Tensor([[6, 7], [8, 9]], gradient=[[-1, -2], [-3, -4]])
        result = a * b
        result.gradient = [[10, -1], [0, 0.5]]
        result.operation(result)
        np.testing.assert_array_equal(a.gradient, [[61, -5], [3, 8.5]])
        np.testing.assert_array_equal(b.gradient, [[19, -5], [-3, -1.5]])
        np.testing.assert_array_equal(result.gradient, [[10, -1], [0, 0.5]])
        np.testing.assert_array_equal(a.value, [[2, 3], [4, 5]])
        np.testing.assert_array_equal(b.value, [[6, 7], [8, 9]])

    def test_repeated_operand_receives_both_local_contributions(self):
        source = Tensor([1, 2, 3], gradient=[7, 11, 13])
        result = source * source
        self.assertIs(result.parents[0], source)
        self.assertIs(result.parents[1], source)
        result.gradient = [0.5, -2, 3]
        result.operation(result)
        np.testing.assert_array_equal(source.gradient, [8, 3, 31])
        np.testing.assert_array_equal(source.value, [1, 2, 3])

    def test_later_operations_supply_the_received_gradient(self):
        a, b = Tensor([2, 3]), Tensor([4, 5])
        weights = Tensor([10, -1])
        product = a * b
        loss = (product * weights).sum()
        backward(loss)
        self.assertEqual(loss.value.item(), 65)
        np.testing.assert_array_equal(product.gradient, [10, -1])
        np.testing.assert_array_equal(a.gradient, [40, -5])
        np.testing.assert_array_equal(b.gradient, [20, -3])
        np.testing.assert_array_equal(weights.gradient, [8, 15])

    def test_both_operands_match_finite_differences_of_a_weighted_loss(self):
        av = np.array([[0.5, -2], [3, -0.25]])
        bv = np.array([[4, 1.5], [-2, 7]])
        weights = np.array([[10, -1], [0.5, -2]])
        a, b = Tensor(av), Tensor(bv)
        loss = ((a * b) * Tensor(weights)).sum()
        backward(loss)
        epsilon = 1e-6
        for values, other, gradient in ((av, bv, a.gradient), (bv, av, b.gradient)):
            expected = np.empty_like(values)
            for index in np.ndindex(values.shape):
                plus, minus = values.copy(), values.copy()
                plus[index] += epsilon
                minus[index] -= epsilon
                expected[index] = (
                    np.sum(weights * plus * other) - np.sum(weights * minus * other)
                ) / (2 * epsilon)
            self.assertEqual(gradient.shape, values.shape)
            np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
