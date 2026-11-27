"""Verify subtraction, multiplication, and division require equal shapes."""

import operator
import unittest

from aurora.tensor import Tensor


class TensorArithmeticShapeTests(unittest.TestCase):
    def test_arithmetic_rejects_broadcastable_but_unequal_shapes(self):
        cases = (([[1, 2], [3, 4]], [1, 2]), ([1, 2], 2), (2, [1, 2]))
        for operation in (operator.sub, operator.mul, operator.truediv):
            for left, right in cases:
                a, b = Tensor(left), Tensor(right)
                with self.subTest(
                    operation=operation.__name__,
                    left=a.value.shape,
                    right=b.value.shape,
                ):
                    with self.assertRaises(ValueError) as error:
                        operation(a, b)
                    self.assertIn(str(a.value.shape), str(error.exception))
                    self.assertIn(str(b.value.shape), str(error.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
