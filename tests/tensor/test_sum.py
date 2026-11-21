"""Full-array sums preserve the graph and propagate scalar gradients."""

import unittest

import numpy as np

from aurora.autograd import backward
from aurora.tensor import Tensor


class TensorSumTests(unittest.TestCase):
    def test_forward_reduces_every_axis_and_retains_the_input_as_parent(self):
        cases = (
            (7.0, 7.0),
            ([1, -2, 4], 3.0),
            ([[1, 2], [3, 4]], 10.0),
            (np.arange(8).reshape(2, 2, 2), 28.0),
            (np.empty((0, 3)), 0.0),
        )
        for values, expected in cases:
            with self.subTest(shape=np.shape(values)):
                source = Tensor(values)
                result = source.sum()
                self.assertIsInstance(result, Tensor)
                self.assertEqual(result.value.shape, ())
                self.assertEqual(result.value.dtype, np.dtype(np.float64))
                self.assertEqual(result.value.item(), expected)
                self.assertEqual(len(result.parents), 1)
                self.assertIs(result.parents[0], source)
                self.assertEqual(result.gradient.shape, ())
                self.assertEqual(result.gradient.item(), 0.0)

    def test_forward_preserves_input_values_and_existing_gradients(self):
        source = Tensor([2, 4, 6], gradient=[10, 20, 30])
        result = source.sum()
        np.testing.assert_array_equal(source.value, [2, 4, 6])
        np.testing.assert_array_equal(source.gradient, [10, 20, 30])
        self.assertFalse(np.shares_memory(source.value, result.value))
        self.assertFalse(np.shares_memory(source.gradient, result.gradient))

    def test_local_rule_accumulates_the_received_scalar_with_input_shape(self):
        for incoming in (3.0, -2.0, 0.0):
            with self.subTest(incoming=incoming):
                source = Tensor([[2, 4], [6, 8]], gradient=[[10, 20], [30, 40]])
                result = source.sum()
                result.gradient = incoming
                result.operation(result)
                self.assertEqual(source.gradient.shape, (2, 2))
                np.testing.assert_array_equal(
                    source.gradient, np.array([[10, 20], [30, 40]]) + incoming
                )
                np.testing.assert_array_equal(source.value, [[2, 4], [6, 8]])
                self.assertEqual(result.gradient.item(), incoming)

    def test_backward_reaches_both_operands_through_addition_and_sum(self):
        for values in (2.0, [1, 2, 3], [[1, 2], [3, 4]]):
            with self.subTest(shape=np.shape(values)):
                a, b = Tensor(values), Tensor(values)
                intermediate = a + b
                loss = intermediate.sum()
                backward(loss)
                for tensor in (a, b, intermediate):
                    self.assertEqual(tensor.gradient.shape, tensor.value.shape)
                    np.testing.assert_array_equal(
                        tensor.gradient, np.ones(tensor.value.shape)
                    )
                self.assertEqual(loss.gradient.shape, ())
                self.assertEqual(loss.gradient.item(), 1.0)

    def test_operation_after_sum_supplies_a_nonunit_incoming_gradient(self):
        source = Tensor([2, 4, 6])
        total = source.sum()
        scale = Tensor(-3.5)
        loss = total * scale
        backward(loss)
        self.assertEqual(loss.value.item(), -42.0)
        self.assertEqual(total.gradient.item(), -3.5)
        np.testing.assert_array_equal(source.gradient, [-3.5, -3.5, -3.5])
        self.assertEqual(scale.gradient.item(), 12.0)

    def test_repeated_addition_operand_receives_both_paths_after_sum(self):
        source = Tensor([1, 2, 3])
        loss = (source + source).sum()
        backward(loss)
        self.assertEqual(loss.value.item(), 12.0)
        np.testing.assert_array_equal(source.gradient, [2, 2, 2])

    def test_composed_array_gradients_match_finite_differences(self):
        values = np.array([[0.5, -2.0], [3.0, -0.25]])
        source = Tensor(values)
        loss = (source * source).sum()
        backward(loss)
        expected = np.empty_like(values)
        epsilon = 1e-6
        for index in np.ndindex(values.shape):
            plus, minus = values.copy(), values.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            expected[index] = (np.sum(plus * plus) - np.sum(minus * minus)) / (
                2 * epsilon
            )
        np.testing.assert_allclose(source.gradient, expected, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
