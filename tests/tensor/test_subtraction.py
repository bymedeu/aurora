"""Verify elementwise subtraction for equal-shaped tensors."""

import unittest

import numpy as np

from aurora.autograd import backward
from aurora.tensor import Tensor


class TensorSubtractionTests(unittest.TestCase):
    def test_forward_preserves_shape_values_and_operand_order(self):
        cases = (
            (2.0, 3.0, -1.0),
            ([2, 4, 6], [1, 5, 6], [1, -1, 0]),
            (
                [[1, 3, 5], [7, 9, 11]],
                [[2, 2, 5], [4, 10, 3]],
                [[-1, 1, 0], [3, -1, 8]],
            ),
        )
        for left, right, expected in cases:
            with self.subTest(shape=np.shape(left)):
                a, b = Tensor(left), Tensor(right)

                difference = a - b

                self.assertEqual(difference.value.shape, np.shape(left))
                np.testing.assert_array_equal(difference.value, expected)
                self.assertEqual(len(difference.parents), 2)
                self.assertIs(difference.parents[0], a)
                self.assertIs(difference.parents[1], b)
                np.testing.assert_array_equal(a.value, left)
                np.testing.assert_array_equal(b.value, right)

    def test_backward_accumulates_opposite_nonuniform_gradients(self):
        a = Tensor([2, 4, 6], gradient=[10, 20, 30])
        b = Tensor([1, 3, 5], gradient=[-10, -20, -30])
        difference = a - b
        weights = Tensor([3, -2, 0.5])
        loss = (difference * weights).sum()

        backward(loss)

        self.assertEqual(loss.value.item(), 1.5)
        np.testing.assert_array_equal(difference.gradient, [3, -2, 0.5])
        np.testing.assert_array_equal(a.gradient, [13, 18, 30.5])
        np.testing.assert_array_equal(b.gradient, [-13, -18, -30.5])

    def test_repeated_operand_contributions_cancel(self):
        a = Tensor([2, 4, 6])
        difference = a - a
        loss = (difference * Tensor([3, -2, 0.5])).sum()

        backward(loss)

        self.assertEqual(loss.value.item(), 0.0)
        np.testing.assert_array_equal(difference.gradient, [3, -2, 0.5])
        np.testing.assert_array_equal(a.gradient, [0, 0, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
