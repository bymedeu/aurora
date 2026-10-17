"""Check the public model's state and connected prediction contract."""

import unittest

from aurora import Aurora
from aurora.autograd import AutoNode, backward


class PredictionTests(unittest.TestCase):
    def test_default_parameters_are_zero_valued_leaves(self):
        model = Aurora()
        for parameter in (model.weight, model.bias):
            self.assertIsInstance(parameter, AutoNode)
            self.assertEqual(parameter.value, 0)
            self.assertEqual(parameter.gradient, 0)
            self.assertEqual(parameter.parents, ())
        self.assertEqual(model.predict(4.0).value, 0)

    def test_predictions_use_the_supplied_parameters_and_input(self):
        model = Aurora(weight=2.0, bias=1.0)
        for value, expected in ((4.0, 9.0), (-1.0, -1.0), (0.0, 1.0)):
            with self.subTest(value=value):
                result = model.predict(value)
                self.assertIsInstance(result, AutoNode)
                self.assertIsInstance(result.value, (float, int))
                self.assertEqual(result.value, expected)

    def test_models_do_not_share_parameter_objects(self):
        first, second = Aurora(), Aurora()
        self.assertIsNot(first.weight, second.weight)
        self.assertIsNot(first.bias, second.bias)
        self.assertIsNot(first.weight, first.bias)
        first.weight.gradient = 7
        self.assertEqual(second.weight.gradient, 0)

    def test_prediction_preserves_parameter_state_and_gradients(self):
        model = Aurora(weight=2.0, bias=1.0)
        weight, bias = model.weight, model.bias
        weight.gradient, bias.gradient = 7, -3

        model.predict(4.0)

        self.assertIs(model.weight, weight)
        self.assertIs(model.bias, bias)
        self.assertEqual((weight.value, bias.value), (2, 1))
        self.assertEqual((weight.gradient, bias.gradient), (7, -3))
        self.assertEqual((weight.parents, bias.parents), ((), ()))

    def test_prediction_backward_reaches_model_parameters(self):
        model = Aurora(weight=2.0, bias=1.0)

        prediction = model.predict(4.0)
        backward(prediction)

        self.assertEqual(model.weight.gradient, 4.0)
        self.assertEqual(model.bias.gradient, 1.0)
