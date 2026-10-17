"""End-to-end backward propagation and shared dependencies."""

import unittest

from aurora.autograd import AutoNode, backward


class BackwardEngineTests(unittest.TestCase):
    def test_fresh_leaf_output_is_seeded(self):
        output = AutoNode(3)
        backward(output)
        self.assertEqual(output.gradient, 1)

    def test_fresh_graph_end_to_end(self):
        a, b = AutoNode(2), AutoNode(3)
        c = a * b
        output = c + a
        backward(output)
        self.assertEqual(output.gradient, 1)
        self.assertEqual(c.gradient, 1)
        self.assertEqual(a.gradient, 4)
        self.assertEqual(b.gradient, 2)
        self.assertEqual((a.value, b.value, c.value, output.value), (2, 3, 6, 8))

    def test_shared_intermediate_receives_both_contributions_before_propagating(self):
        a, b = AutoNode(2), AutoNode(3)
        shared = a * b
        left = shared + a
        right = shared * b
        output = left + right
        # Isolate traversal from the separate automatic-seeding contract.
        output.gradient = 1
        backward(output)
        # output = a*b + a + a*b*b
        self.assertEqual(shared.gradient, 4)
        self.assertEqual(a.gradient, 13)
        self.assertEqual(b.gradient, 14)
        self.assertEqual((left.gradient, right.gradient), (1, 1))

    def test_repeated_intermediate_is_propagated_once(self):
        a = AutoNode(3)
        square = a * a
        output = square + square
        output.gradient = 1
        backward(output)
        self.assertEqual(square.gradient, 2)
        self.assertEqual(a.gradient, 12)

    def test_composed_rules_match_finite_differences(self):
        def expression(a, b):
            return ((a * b) + a) / (b - a)

        epsilon = 1e-6
        for av, bv in ((2.0, 4.0), (-1.5, 2.0), (0.0, 3.0)):
            with self.subTest(a=av, b=bv):
                a, b = AutoNode(av), AutoNode(bv)
                output = expression(a, b)
                output.gradient = 1
                backward(output)
                expected_a = (
                    expression(av + epsilon, bv) - expression(av - epsilon, bv)
                ) / (2 * epsilon)
                expected_b = (
                    expression(av, bv + epsilon) - expression(av, bv - epsilon)
                ) / (2 * epsilon)
                self.assertAlmostEqual(a.gradient, expected_a, delta=1e-6)
                self.assertAlmostEqual(b.gradient, expected_b, delta=1e-6)

    def test_fresh_graphs_accumulate_into_shared_leaf(self):
        a = AutoNode(3)
        first = a * a
        first.gradient = 1
        backward(first)
        self.assertEqual(a.gradient, 6)
        # Fresh intermediates; this does not test reusing an old graph.
        second = a + a
        second.gradient = 1
        backward(second)
        self.assertEqual(a.gradient, 8)

