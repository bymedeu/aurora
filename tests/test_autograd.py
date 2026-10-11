"""Checks for local derivative rules, graph ordering, and backward execution."""

import math
import unittest

from aurora.autograd import AutoNode, backward, clear_gradients
from aurora.autograd.engine import _topological_order


class ScalarNodeTests(unittest.TestCase):
    def test_forward_values_and_parent_identity(self):
        a, b = AutoNode(2), AutoNode(3)
        c = a * b
        d = c + a
        self.assertEqual(c.value, 6)
        self.assertEqual(d.value, 8)
        self.assertIs(c.parents[0], a)
        self.assertIs(c.parents[1], b)
        self.assertIs(d.parents[0], c)
        self.assertIs(d.parents[1], a)
        self.assertEqual(a.parents, ())
        self.assertTrue(all(n.gradient == 0 for n in (a, b, c, d)))

    def test_local_rules_accumulate_scaled_contributions(self):
        # At a=2, b=4, these are the analytical local derivatives.
        cases = (
            ("addition", lambda a, b: a + b, (1, 1)),
            ("subtraction", lambda a, b: a - b, (1, -1)),
            ("multiplication", lambda a, b: a * b, (4, 2)),
            ("division", lambda a, b: a / b, (0.25, -0.125)),
        )
        for name, operation, derivatives in cases:
            for incoming in (5, 0, -2):
                with self.subTest(operation=name, incoming=incoming):
                    a, b = AutoNode(2), AutoNode(4)
                    a.gradient, b.gradient = 7, 11
                    result = operation(a, b)
                    result.gradient = incoming
                    result.operation(result)
                    self.assertAlmostEqual(a.gradient, 7 + incoming * derivatives[0])
                    self.assertAlmostEqual(b.gradient, 11 + incoming * derivatives[1])
                    self.assertEqual((a.value, b.value), (2, 4))

    def test_repeated_operand_contributions(self):
        cases = (
            ("addition", lambda a: a + a, 2),
            ("subtraction", lambda a: a - a, 0),
            ("multiplication", lambda a: a * a, 6),
            ("division", lambda a: a / a, 0),
        )
        for name, operation, expected in cases:
            with self.subTest(operation=name):
                a = AutoNode(3)
                result = operation(a)
                result.gradient = 1
                result.operation(result)
                self.assertAlmostEqual(a.gradient, expected)

    def test_leaf_no_op_and_independent_gradients(self):
        a, b = AutoNode(2), AutoNode(2)
        a.gradient = 5
        a.operation(a)
        self.assertEqual(a.gradient, 5)
        self.assertEqual(b.gradient, 0)
        self.assertEqual(a.value, 2)

    def test_zero_numerator_and_zero_denominator(self):
        a, b = AutoNode(0), AutoNode(4)
        result = a / b
        result.gradient = 5
        result.operation(result)
        self.assertAlmostEqual(a.gradient, 1.25)
        self.assertAlmostEqual(b.gradient, 0)
        with self.assertRaises(ZeroDivisionError):
            b / a  # type: ignore

    def test_value_is_read_only_while_gradient_remains_mutable(self):
        a = AutoNode(3)
        self.assertEqual(a.value, 3)
        with self.assertRaises(AttributeError):
            a.value = 8  # type: ignore
        self.assertEqual(a.value, 3)
        a.gradient = 67
        self.assertEqual(a.gradient, 67)


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


class SigmoidTests(unittest.TestCase):
    def test_forward_value_and_parent_connection(self):
        z = AutoNode(0)
        result = z.sigmoid()
        self.assertIsInstance(result, AutoNode)
        self.assertIsNot(result, z)
        self.assertEqual(result.value, 0.5)
        self.assertEqual(result.gradient, 0)
        self.assertEqual(len(result.parents), 1)
        self.assertIs(result.parents[0], z)

    def test_local_rule_scales_and_accumulates_incoming_gradient(self):
        for incoming in (0, 1, 8, -3):
            with self.subTest(incoming=incoming):
                z = AutoNode(0)
                result = z.sigmoid()
                z.gradient = 7
                result.gradient = incoming
                result.operation(result)
                self.assertEqual(z.gradient, 7 + incoming * 0.25)
                self.assertEqual((z.value, result.value), (0, 0.5))

    def test_shared_sigmoid_result_receives_all_contributions(self):
        z = AutoNode(0)
        s = z.sigmoid()
        output = s * s + s
        backward(output)
        self.assertEqual(s.gradient, 2)
        self.assertEqual(z.gradient, 0.5)

    def test_composed_gradient_matches_finite_differences(self):
        def numeric_loss(z):
            s = 0.5 * (1 + math.tanh(z / 2))
            return (s - 0.3) ** 2

        epsilon = 1e-6
        for value in (-3.0, -0.5, 0.0, 1.5, 3.0):
            with self.subTest(value=value):
                z = AutoNode(value)
                error = z.sigmoid() - AutoNode(0.3)
                loss = error * error
                backward(loss)
                expected = (
                    numeric_loss(value + epsilon) - numeric_loss(value - epsilon)
                ) / (2 * epsilon)
                self.assertAlmostEqual(z.gradient, expected, delta=1e-8)

    def test_large_finite_inputs_have_finite_outputs_and_gradients(self):
        for value, expected in ((-1000.0, 0.0), (1000.0, 1.0)):
            with self.subTest(value=value):
                z = AutoNode(value)
                result = z.sigmoid()
                self.assertTrue(math.isfinite(result.value))
                self.assertEqual(result.value, expected)
                backward(result)
                self.assertTrue(math.isfinite(z.gradient))
                self.assertAlmostEqual(z.gradient, 0.0, delta=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
