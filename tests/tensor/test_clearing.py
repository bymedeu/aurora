"""Shape-preserving gradient clearing and unchanged tensor graph reuse."""

import unittest

import numpy as np

from aurora.autograd import backward, clear_gradients
from aurora.tensor import Tensor


class TensorClearingTests(unittest.TestCase):
    def test_clears_every_reachable_gradient_with_its_value_shape(self):
        cases = (2.0, [1, 2, 3], [[1, 2], [3, 4]], np.empty((0, 3)))
        for values in cases:
            with self.subTest(shape=np.shape(values)):
                source = Tensor(values)
                intermediate = source + source
                loss = intermediate.sum()
                backward(loss)
                clear_gradients(loss)
                for node in (source, intermediate, loss):
                    self.assertIsInstance(node.gradient, np.ndarray)
                    self.assertEqual(node.gradient.shape, node.value.shape)
                    np.testing.assert_array_equal(
                        node.gradient, np.zeros(node.value.shape)
                    )

    def test_clearing_preserves_values_graph_and_unreachable_gradients(self):
        a, b = Tensor([[1, 2], [3, 4]]), Tensor([[5, 6], [7, 8]])
        intermediate = a + b
        loss = intermediate.sum()
        nodes = (a, b, intermediate, loss)
        saved = [(node.value.copy(), node.parents, node.operation) for node in nodes]
        unrelated = Tensor([7, 8], gradient=[9, 10])
        backward(loss)
        clear_gradients(loss)
        for node, (value, parents, operation) in zip(nodes, saved):
            np.testing.assert_array_equal(node.value, value)
            self.assertIs(node.parents, parents)
            self.assertIs(node.operation, operation)
        np.testing.assert_array_equal(unrelated.value, [7, 8])
        np.testing.assert_array_equal(unrelated.gradient, [9, 10])

    def test_repeated_clearing_and_backward_reproduce_nonuniform_gradients(self):
        source = Tensor([[0.5, -2], [3, -0.25]])
        square = source * source
        loss = square.sum()
        nodes = (source, square, loss)
        backward(loss)
        np.testing.assert_array_equal(source.gradient, [[1, -4], [6, -0.5]])
        expected = [node.gradient.copy() for node in nodes]
        for _ in range(2):
            clear_gradients(loss)
            clear_gradients(loss)
            for node in nodes:
                self.assertEqual(node.gradient.shape, node.value.shape)
                np.testing.assert_array_equal(node.gradient, np.zeros(node.value.shape))
            backward(loss)
            for node, gradient in zip(nodes, expected):
                self.assertEqual(node.gradient.shape, node.value.shape)
                np.testing.assert_array_equal(node.gradient, gradient)


if __name__ == "__main__":
    unittest.main(verbosity=2)
