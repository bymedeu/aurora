"""Explicit clearing and unchanged-graph reuse."""

import unittest

from aurora.autograd import AutoNode, backward, clear_gradients


class GradientClearingTests(unittest.TestCase):
    def test_clears_reachable_gradients_and_preserves_graph(self):
        a, b = AutoNode(2), AutoNode(3)
        shared = a * b
        output = shared + shared
        unrelated = AutoNode(7)
        unrelated.gradient = 19
        nodes = (a, b, shared, output)
        saved = [(node.value, node.parents, node.operation) for node in nodes]
        backward(output)

        clear_gradients(output)

        for node, (value, parents, operation) in zip(nodes, saved):
            self.assertEqual(node.gradient, 0)
            self.assertEqual(node.value, value)
            self.assertIs(node.parents, parents)
            self.assertIs(node.operation, operation)
        self.assertEqual(unrelated.gradient, 19)

    def test_backward_after_clearing_reproduces_shared_graph_gradients(self):
        a = AutoNode(3)
        square = a * a
        output = square + square
        backward(output)
        self.assertEqual((a.gradient, square.gradient, output.gradient), (12, 2, 1))

        for _ in range(2):
            clear_gradients(output)
            backward(output)
            self.assertEqual((a.gradient, square.gradient, output.gradient), (12, 2, 1))

    def test_leaf_output_and_repeated_clearing(self):
        leaf = AutoNode(4)
        backward(leaf)
        clear_gradients(leaf)
        self.assertEqual(leaf.gradient, 0)
        clear_gradients(leaf)
        self.assertEqual(leaf.gradient, 0)
        self.assertEqual(leaf.value, 4)
        backward(leaf)
        self.assertEqual(leaf.gradient, 1)

