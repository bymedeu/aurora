"""Check nonlinear forward values, all seven gradients, and the training caller."""

import math
from pathlib import Path
import runpy
import unittest

from aurora.autograd import AutoNode, backward


EXAMPLE = Path(__file__).resolve().parents[2] / "examples/autograd/nonlinear.py"
forward = runpy.run_path(str(EXAMPLE))["forward"]


def numeric_prediction(value, parameters):
    w1, b1, w2, b2, v1, v2, bias = parameters
    h1 = 1.0 / (1.0 + math.exp(-(w1 * value + b1)))
    h2 = 1.0 / (1.0 + math.exp(-(w2 * value + b2)))
    return v1 * h1 + v2 * h2 + bias


class NonlinearForwardTests(unittest.TestCase):
    def test_forward_matches_the_agreed_numerical_checkpoint(self):
        parameters = [AutoNode(value) for value in (1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0)]

        prediction = forward(AutoNode(1.0), parameters)

        self.assertIsInstance(prediction, AutoNode)
        self.assertIsInstance(prediction.value, (float, int))
        self.assertAlmostEqual(prediction.value, 0.4621171572600098)
        self.assertEqual([parameter.gradient for parameter in parameters], [0] * 7)

    def test_all_parameter_gradients_match_finite_differences(self):
        epsilon = 1e-6
        parameter_names = ("w1", "b1", "w2", "b2", "v1", "v2", "c")
        cases = (
            (1.0, (1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0)),
            (-0.7, (0.3, -0.2, -0.8, 0.4, 1.2, -0.6, 0.1)),
        )
        for value, initial_values in cases:
            with self.subTest(input=value):
                parameters = [AutoNode(number) for number in initial_values]
                prediction = forward(AutoNode(value), parameters)
                backward(prediction)

                for index, name in enumerate(parameter_names):
                    with self.subTest(parameter=name):
                        plus, minus = list(initial_values), list(initial_values)
                        plus[index] += epsilon
                        minus[index] -= epsilon
                        expected = (
                            numeric_prediction(value, plus)
                            - numeric_prediction(value, minus)
                        ) / (2 * epsilon)
                        self.assertAlmostEqual(parameters[index].gradient, expected, delta=1e-8)
                self.assertEqual(tuple(parameter.value for parameter in parameters), initial_values)

    def test_training_example_rebuilds_batches_and_reduces_loss(self):
        state = runpy.run_path(str(EXAMPLE), run_name="__main__")
        parameters = state["parameters"]
        inputs, targets = state["inputs"], state["targets"]
        initial_values = (1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0)

        self.assertGreater(state["steps"], 1)
        self.assertEqual(len(parameters), 7)
        self.assertEqual(len(state["preds"]), len(inputs))
        for parameter in parameters:
            self.assertIsInstance(parameter, AutoNode)
            self.assertTrue(math.isfinite(parameter.value))
            self.assertEqual(parameter.parents, ())
            self.assertEqual(parameter.gradient, 0)

        # Evaluate the retained parameters independently of the last pre-update loss.
        final_values = [parameter.value for parameter in parameters]
        initial_loss = sum(
            (numeric_prediction(x, initial_values) - y) ** 2
            for x, y in zip(inputs, targets)
        ) / len(inputs)
        final_loss = sum(
            (numeric_prediction(x, final_values) - y) ** 2
            for x, y in zip(inputs, targets)
        ) / len(inputs)
        self.assertTrue(math.isfinite(final_loss))
        self.assertLess(final_loss, initial_loss / 10)
