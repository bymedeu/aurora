"""Graph discovery, dependency ordering, and cycle rejection."""

import unittest

from aurora.autograd import AutoNode, backward
from aurora.autograd.engine import _topological_order


class GraphOrderingTests(unittest.TestCase):
    def assert_valid_order(self, output, expected_nodes):
        order = _topological_order(output)
        identities = [id(node) for node in order]
        self.assertEqual(len(identities), len(set(identities)), "Duplicate node")
        self.assertEqual(set(identities), {id(node) for node in expected_nodes})
        self.assertIs(order[-1], output)
        positions = {identity: index for index, identity in enumerate(identities)}
        for node in order:
            for parent in node.parents:
                self.assertLess(positions[id(parent)], positions[id(node)])
        # Discovery must not execute derivative rules or seed gradients.
        self.assertTrue(all(node.gradient == 0 for node in expected_nodes))

    def test_shared_leaf(self):
        a, b = AutoNode(2), AutoNode(3)
        c = a * b
        d = c + a
        self.assert_valid_order(d, (a, b, c, d))

    def test_shared_intermediate_and_equal_valued_distinct_nodes(self):
        a, b = AutoNode(2), AutoNode(2)
        shared = a * b
        left = shared + a
        right = shared + b
        output = left * right
        self.assert_valid_order(output, (a, b, shared, left, right, output))

    def test_repeated_operand_and_single_leaf(self):
        a = AutoNode(3)
        squared = a * a
        self.assert_valid_order(squared, (a, squared))
        self.assert_valid_order(a, (a,))

    def test_cycle_is_rejected_before_backward_mutates_gradients(self):
        output = AutoNode(3)
        output.parents = (output,)
        output.gradient = 7

        with self.assertRaisesRegex(ValueError, "cycle"):
            backward(output)

        self.assertEqual(output.gradient, 7)
