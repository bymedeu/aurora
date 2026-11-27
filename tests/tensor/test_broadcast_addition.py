"""Check broadcast addition and reduction to each operand's original shape."""

import unittest

import numpy as np

from aurora.autograd import backward
from aurora.tensor import Tensor


class TensorBroadcastAdditionTests(unittest.TestCase):
    def test_square_matrix_and_vector_align_on_the_last_axis(self):
        for vector_first in (False, True):
            with self.subTest(vector_first=vector_first):
                matrix = Tensor([[1, 2], [3, 4]])
                vector = Tensor([10, 20])
                weights = Tensor([[1, 2], [3, 4]])

                result = vector + matrix if vector_first else matrix + vector
                backward((result * weights).sum())

                np.testing.assert_array_equal(result.value, [[11, 22], [13, 24]])
                np.testing.assert_array_equal(matrix.gradient, [[1, 2], [3, 4]])
                np.testing.assert_array_equal(vector.gradient, [4, 6])

    def test_both_operands_reduce_the_axes_where_they_were_broadcast(self):
        a = Tensor(np.arange(8).reshape(2, 1, 4))
        b = Tensor([[10], [20], [30]])
        weights = Tensor(np.arange(1, 25).reshape(2, 3, 4))

        result = a + b
        backward((result * weights).sum())

        self.assertEqual(result.value.shape, (2, 3, 4))
        np.testing.assert_array_equal(result.value, a.value + b.value)
        np.testing.assert_array_equal(
            a.gradient, [[[15, 18, 21, 24]], [[51, 54, 57, 60]]]
        )
        np.testing.assert_array_equal(b.gradient, [[68], [100], [132]])

    def test_scalar_tensor_accumulates_contributions_from_every_element(self):
        array = Tensor([[1, 2], [3, 4]])
        scalar = Tensor(5, gradient=10)
        weights = Tensor([[1, -2], [3, 4]])

        result = array + scalar
        backward((result * weights).sum())

        np.testing.assert_array_equal(result.value, [[6, 7], [8, 9]])
        np.testing.assert_array_equal(array.gradient, [[1, -2], [3, 4]])
        self.assertEqual(scalar.gradient.shape, ())
        self.assertEqual(scalar.gradient.item(), 16)

    def test_equal_element_counts_do_not_make_shapes_compatible(self):
        a = Tensor(np.ones((2, 3)))
        b = Tensor(np.ones((3, 2)))

        with self.assertRaises(ValueError):
            _ = a + b


if __name__ == "__main__":
    unittest.main(verbosity=2)
