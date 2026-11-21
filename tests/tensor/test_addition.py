"""Check equal-shape addition, local accumulation, and a scalar backward path."""

import unittest

import numpy as np

from aurora.autograd import backward
from aurora.tensor import Tensor


class TensorAdditionTests(unittest.TestCase):
    def test_equal_shape_addition_preserves_values_shape_and_parent_identity(self):
        cases = (
            (2.0, -3.0, -1.0),
            ([1, -2, 3], [4, 5, -6], [5, 3, -3]),
            (
                [[1, 2, 3], [4, 5, 6]],
                [[-1, 2, -3], [1, -2, 3]],
                [[0, 4, 0], [5, 3, 9]],
            ),
        )
        for left, right, expected in cases:
            with self.subTest(shape=np.shape(left)):
                a, b = Tensor(left), Tensor(right)

                result = a + b

                self.assertIsInstance(result, Tensor)
                self.assertEqual(result.value.shape, a.value.shape)
                self.assertEqual(result.value.dtype, np.dtype(np.float64))
                np.testing.assert_array_equal(result.value, expected)
                self.assertEqual(len(result.parents), 2)
                self.assertIs(result.parents[0], a)
                self.assertIs(result.parents[1], b)
                self.assertTrue(callable(result.operation))
                self.assertEqual(result.gradient.shape, result.value.shape)
                np.testing.assert_array_equal(
                    result.gradient, np.zeros(result.value.shape)
                )

    def test_repeated_operand_is_retained_in_both_parent_positions(self):
        tensor = Tensor([1.0, -2.0])

        result = tensor + tensor

        self.assertEqual(len(result.parents), 2)
        self.assertIs(result.parents[0], tensor)
        self.assertIs(result.parents[1], tensor)
        np.testing.assert_array_equal(result.value, [2, -4])

    def test_forward_addition_preserves_existing_input_values_and_gradients(self):
        a, b = Tensor([1.0, 2.0]), Tensor([3.0, 4.0])
        a.gradient[...] = [7.0, -3.0]
        b.gradient[...] = [2.0, 5.0]

        result = a + b

        np.testing.assert_array_equal(a.value, [1, 2])
        np.testing.assert_array_equal(b.value, [3, 4])
        np.testing.assert_array_equal(a.gradient, [7, -3])
        np.testing.assert_array_equal(b.gradient, [2, 5])
        self.assertFalse(np.shares_memory(result.value, a.value))
        self.assertFalse(np.shares_memory(result.value, b.value))

    def test_local_backward_accumulates_nonuniform_incoming_gradients(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        a.gradient[...] = [[7, 11], [13, 17]]
        b.gradient[...] = [[-1, 2], [3, -4]]
        result = a + b
        result.gradient[...] = [[0.5, -2], [3, 0]]

        result.operation(result)

        np.testing.assert_array_equal(a.gradient, [[7.5, 9], [16, 17]])
        np.testing.assert_array_equal(b.gradient, [[-0.5, 0], [6, -4]])
        np.testing.assert_array_equal(a.value, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(b.value, [[5, 6], [7, 8]])
        np.testing.assert_array_equal(result.gradient, [[0.5, -2], [3, 0]])

    def test_local_backward_adds_both_contributions_for_a_repeated_operand(self):
        tensor = Tensor([1.0, 2.0, 3.0])
        tensor.gradient[...] = [7, 11, 13]
        result = tensor + tensor
        result.gradient[...] = [0.5, -2, 3]

        result.operation(result)

        np.testing.assert_array_equal(tensor.gradient, [8, 7, 19])
        np.testing.assert_array_equal(tensor.value, [1, 2, 3])

    def test_scalar_addition_supports_the_existing_backward_entry_point(self):
        a, b = Tensor(2.0), Tensor(-3.0)
        result = a + b

        backward(result)

        for tensor in (a, b, result):
            self.assertIsInstance(tensor.gradient, np.ndarray)
            self.assertEqual(tensor.gradient.shape, ())
            self.assertEqual(tensor.gradient.item(), 1.0)
        self.assertEqual(result.value.item(), -1.0)
