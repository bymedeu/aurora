"""Behavioral checkpoints for the owner-implemented training method.

These are normal tests: unfinished fit behavior must fail visibly, not be skipped.
"""

import unittest

from aurora import Aurora
from aurora.autograd import AutoNode


INPUTS = [-2.0, -1.0, 0.0, 1.0, 2.0]
TARGETS = [-3.0, -1.0, 1.0, 3.0, 5.0]


class FitTests(unittest.TestCase):
    def test_one_step_changes_parameters_and_lowers_loss(self):
        model = Aurora()

        result = model.fit(INPUTS, TARGETS, steps=1, learning_rate=0.1)

        self.assertIsNone(result)
        self.assertAlmostEqual(model.weight.value, 0.8)
        self.assertAlmostEqual(model.bias.value, 0.2)
        self.assertAlmostEqual(model.predict(4.0).value, 3.4)
        loss = sum((model.predict(x).value - y) ** 2 for x, y in zip(INPUTS, TARGETS)) / 5
        self.assertAlmostEqual(loss, 3.52)

    def test_updated_parameters_are_fresh_leaves(self):
        model = Aurora()
        old_weight, old_bias = model.weight, model.bias

        model.fit(INPUTS, TARGETS, steps=1, learning_rate=0.1)

        for old, new in ((old_weight, model.weight), (old_bias, model.bias)):
            self.assertIsInstance(new, AutoNode)
            self.assertIsNot(new, old)
            self.assertEqual(new.parents, ())
            self.assertEqual(new.gradient, 0)
        self.assertEqual((old_weight.value, old_bias.value), (0, 0))

    def test_supplied_initial_state_and_learning_rate_are_used(self):
        model = Aurora(weight=1.0, bias=-1.0)

        model.fit([0.0, 2.0], [1.0, 5.0], steps=1, learning_rate=0.05)

        self.assertAlmostEqual(model.weight.value, 1.4)
        self.assertAlmostEqual(model.bias.value, -0.7)

    def test_two_steps_recompute_predictions_and_gradients(self):
        model = Aurora()

        model.fit(INPUTS, TARGETS, steps=2, learning_rate=0.1)

        self.assertAlmostEqual(model.weight.value, 1.28)
        self.assertAlmostEqual(model.bias.value, 0.36)

    def test_fifty_steps_learn_and_predict_for_a_new_input(self):
        model = Aurora()

        model.fit(INPUTS, TARGETS, steps=50, learning_rate=0.1)

        self.assertAlmostEqual(model.weight.value, 2.0, delta=1e-8)
        self.assertAlmostEqual(model.bias.value, 1.0, delta=2e-5)
        self.assertAlmostEqual(model.predict(4.0).value, 9.0, delta=2e-5)

    def test_training_uses_the_supplied_targets(self):
        model = Aurora()
        different_targets = [-3 * x + 2 for x in INPUTS]

        model.fit(INPUTS, different_targets, steps=50, learning_rate=0.1)

        self.assertAlmostEqual(model.weight.value, -3.0, delta=1e-8)
        self.assertAlmostEqual(model.bias.value, 2.0, delta=4e-5)
        self.assertAlmostEqual(model.predict(4.0).value, -10.0, delta=4e-5)

    def test_separate_fit_calls_continue_from_current_state(self):
        split, combined = Aurora(), Aurora()

        split.fit(INPUTS, TARGETS, steps=1, learning_rate=0.1)
        split.fit(INPUTS, TARGETS, steps=1, learning_rate=0.1)
        combined.fit(INPUTS, TARGETS, steps=2, learning_rate=0.1)

        self.assertAlmostEqual(split.weight.value, 1.28)
        self.assertAlmostEqual(split.bias.value, 0.36)
        self.assertAlmostEqual(split.weight.value, combined.weight.value)
        self.assertAlmostEqual(split.bias.value, combined.bias.value)

    def test_zero_steps_preserve_the_existing_parameters(self):
        model = Aurora(weight=2.0, bias=1.0)
        weight, bias = model.weight, model.bias

        result = model.fit(INPUTS, TARGETS, steps=0, learning_rate=0.1)

        self.assertIsNone(result)
        self.assertIs(model.weight, weight)
        self.assertIs(model.bias, bias)
        self.assertEqual(model.predict(4.0).value, 9.0)

    def test_single_sample_still_updates_both_parameters(self):
        model = Aurora()

        model.fit([2.0], [5.0], steps=1, learning_rate=0.1)

        self.assertAlmostEqual(model.weight.value, 2.0)
        self.assertAlmostEqual(model.bias.value, 1.0)
        self.assertAlmostEqual(model.predict(2.0).value, 5.0)
