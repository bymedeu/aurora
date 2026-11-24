"""Non-empty full-array means and their accumulated graph gradients."""

import unittest

import numpy as np

from aurora.autograd import backward
from aurora.tensor import Tensor


class TensorMeanTests(unittest.TestCase):
    def test_forward_returns_a_scalar_average_for_every_input_rank(self):
        cases = (
            (7.0, 7.0),
            ([1, -2, 4], 1.0),
            ([[2, 4], [6, 8]], 5.0),
            ([[1, 2, 3], [4, 5, 6]], 3.5),
            (np.arange(8).reshape(2, 2, 2), 3.5),
        )
        for values, expected in cases:
            with self.subTest(shape=np.shape(values)):
                result = Tensor(values).mean()
                self.assertIsInstance(result, Tensor)
                self.assertEqual(result.value.shape, ())
                self.assertEqual(result.value.dtype, np.dtype(np.float64))
                self.assertEqual(result.value.item(), expected)
                self.assertEqual(result.gradient.shape, ())
                self.assertEqual(result.gradient.item(), 0.0)

    def test_forward_preserves_input_values_and_existing_gradients(self):
        source = Tensor([2, 4, 6], gradient=[10, 20, 30])
        result = source.mean()
        np.testing.assert_array_equal(source.value, [2, 4, 6])
        np.testing.assert_array_equal(source.gradient, [10, 20, 30])
        self.assertFalse(np.shares_memory(source.value, result.value))
        self.assertFalse(np.shares_memory(source.gradient, result.gradient))

    def test_backward_distributes_unit_gradient_using_total_element_count(self):
        for values in (7.0, [1, 2, 3], np.arange(6).reshape(2, 3)):
            with self.subTest(shape=np.shape(values)):
                source = Tensor(values)
                loss = source.mean()
                backward(loss)
                self.assertEqual(source.gradient.shape, source.value.shape)
                np.testing.assert_allclose(
                    source.gradient, np.full(source.value.shape, 1 / np.size(values))
                )
                self.assertEqual(loss.gradient.item(), 1.0)

    def test_later_operation_supplies_a_scaled_gradient_that_accumulates(self):
        values = np.array([[2, 4], [6, 8]])
        initial = np.array([[10, 20], [30, 40]])
        for incoming in (3.0, -2.0, 0.0):
            with self.subTest(incoming=incoming):
                source = Tensor(values, gradient=initial)
                average, scale = source.mean(), Tensor(incoming)
                loss = average * scale
                backward(loss)
                self.assertEqual(average.gradient.item(), incoming)
                self.assertEqual(scale.gradient.item(), 5.0)
                self.assertEqual(source.gradient.shape, values.shape)
                np.testing.assert_allclose(source.gradient, initial + incoming / 4)
                np.testing.assert_array_equal(source.value, values)

    def test_backward_reaches_both_operands_through_addition(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = Tensor([[6, 5, 4], [3, 2, 1]])
        intermediate = a + b
        loss = intermediate.mean()
        backward(loss)
        self.assertEqual(loss.value.item(), 7.0)
        for tensor in (a, b, intermediate):
            self.assertEqual(tensor.gradient.shape, (2, 3))
            np.testing.assert_allclose(tensor.gradient, np.full((2, 3), 1 / 6))

    def test_repeated_operand_receives_both_paths_after_mean(self):
        source = Tensor([1, 2, 3, 4])
        loss = (source + source).mean()
        backward(loss)
        self.assertEqual(loss.value.item(), 5.0)
        np.testing.assert_array_equal(source.gradient, [0.5, 0.5, 0.5, 0.5])

    def test_composed_gradients_match_finite_differences(self):
        values = np.array([[0.5, -2, 3], [-0.25, 1.5, -4]])
        source = Tensor(values)
        loss = (source * source).mean()
        backward(loss)
        expected = np.empty_like(values)
        epsilon = 1e-6
        for index in np.ndindex(values.shape):
            plus, minus = values.copy(), values.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            expected[index] = (np.mean(plus * plus) - np.mean(minus * minus)) / (
                2 * epsilon
            )
        self.assertEqual(source.gradient.shape, values.shape)
        np.testing.assert_allclose(source.gradient, expected, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
