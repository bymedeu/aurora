"""Scalar values, arithmetic, and local derivative rules."""

import unittest

from aurora.autograd import AutoNode


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

