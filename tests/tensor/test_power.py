"""Verify elementwise integer powers and their gradients."""

import unittest

import numpy as np

from aurora.autograd import backward
from aurora.tensor import Tensor


class TensorPowerTests(unittest.TestCase):
    def test_positive_powers_preserve_shape_and_match_numpy(self):
        for values in (3.0, [2, -3], [[2, -3, 0], [4, 1, -2]]):
            for exponent in (1, 2, 5):
                with self.subTest(shape=np.shape(values), exponent=exponent):
                    base = Tensor(values)

                    result = base**exponent

                    self.assertIsInstance(result, Tensor)
                    self.assertEqual(result.value.shape, base.value.shape)
                    np.testing.assert_array_equal(
                        result.value, np.power(values, exponent)
                    )
                    np.testing.assert_array_equal(base.value, values)

    def test_zero_power_returns_shaped_ones_and_preserves_input_gradient(self):
        base = Tensor([[2, -3], [0, 4]], gradient=[[1, 2], [3, 4]])

        result = base**0
        backward(result.sum())

        self.assertIsInstance(result, Tensor)
        np.testing.assert_array_equal(result.value, np.ones((2, 2)))
        np.testing.assert_array_equal(base.gradient, [[1, 2], [3, 4]])

    def test_weighted_loss_accumulates_power_gradients(self):
        base = Tensor([2, -3, 0], gradient=[1, 2, 3])
        weights = Tensor([0.5, -2, 3])

        backward(((base**5) * weights).sum())

        np.testing.assert_array_equal(base.gradient, [41, -808, 3])

    def test_invalid_exponents_raise(self):
        base = Tensor([2, 3])
        for exponent, error in ((-1, ValueError), (1.5, TypeError)):
            with self.subTest(exponent=exponent), self.assertRaises(error):
                _ = base**exponent


if __name__ == "__main__":
    unittest.main(verbosity=2)
