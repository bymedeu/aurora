"""Check normalized values, gradient storage, and tensor ownership contracts."""

import unittest

import numpy as np

from aurora.tensor import Tensor


class TensorConstructionTests(unittest.TestCase):
    def test_numerical_scalars_become_float64_zero_dimensional_arrays(self):
        for supplied in (3, -2.5, np.float32(0.25)):
            with self.subTest(input=supplied):
                tensor = Tensor(supplied)
                self.assertIsInstance(tensor.value, np.ndarray)
                self.assertEqual(tensor.value.shape, ())
                self.assertEqual(tensor.value.dtype, np.dtype(np.float64))
                self.assertEqual(tensor.value.item(), float(supplied))

    def test_list_becomes_a_vector_without_changing_its_numbers(self):
        tensor = Tensor([1, -2, 3])

        self.assertIsInstance(tensor.value, np.ndarray)
        self.assertEqual(tensor.value.shape, (3,))
        self.assertEqual(tensor.value.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(tensor.value, [1.0, -2.0, 3.0])

    def test_rectangular_nested_list_preserves_rows_and_columns(self):
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(tensor.value.shape, (2, 3))
        self.assertEqual(tensor.value.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(tensor.value, [[1, 2, 3], [4, 5, 6]])

    def test_higher_dimensional_array_keeps_its_axes(self):
        supplied = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

        tensor = Tensor(supplied)

        self.assertEqual(tensor.value.shape, (2, 3, 4))
        np.testing.assert_array_equal(tensor.value, supplied)

    def test_integer_and_float_arrays_use_float64_storage(self):
        for dtype in (np.int32, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                supplied = np.array([[1, -2], [3, 4]], dtype=dtype)
                tensor = Tensor(supplied)

                self.assertEqual(tensor.value.dtype, np.dtype(np.float64))
                self.assertEqual(tensor.value.shape, supplied.shape)
                np.testing.assert_array_equal(tensor.value, supplied)
                self.assertEqual(supplied.dtype, np.dtype(dtype))

    def test_noncontiguous_input_is_independent_of_its_source_storage(self):
        base = np.arange(24, dtype=np.float64).reshape(4, 6)
        supplied = base[::2, 1::2].T
        expected = np.array([[1, 13], [3, 15], [5, 17]], dtype=np.float64)

        tensor = Tensor(supplied)
        base[...] = -99.0

        self.assertEqual(tensor.value.shape, (3, 2))
        np.testing.assert_array_equal(tensor.value, expected)
        self.assertFalse(np.shares_memory(tensor.value, base))

    def test_modifying_original_array_does_not_change_the_tensor(self):
        supplied = np.array([[1.0, 2.0], [3.0, 4.0]])

        tensor = Tensor(supplied)
        supplied[0, 0] = 99.0

        np.testing.assert_array_equal(tensor.value, [[1, 2], [3, 4]])
        self.assertFalse(np.shares_memory(tensor.value, supplied))

    def test_default_gradient_is_zero_float64_storage_matching_each_value_shape(self):
        inputs = (
            2.0,
            [1, 2, 3],
            [[1, 2], [3, 4]],
            np.arange(8).reshape(2, 2, 2),
        )
        for supplied in inputs:
            with self.subTest(input_shape=np.shape(supplied)):
                tensor = Tensor(supplied)

                self.assertIsInstance(tensor.gradient, np.ndarray)
                self.assertEqual(tensor.gradient.shape, tensor.value.shape)
                self.assertEqual(tensor.gradient.dtype, np.dtype(np.float64))
                np.testing.assert_array_equal(
                    tensor.gradient, np.zeros(tensor.value.shape)
                )

    def test_default_gradients_are_writable_and_do_not_share_value_or_other_gradients(
        self,
    ):
        supplied = np.array([1.0, 2.0])
        first, second = Tensor(supplied), Tensor(supplied)

        first.gradient[...] = [7.0, -3.0]

        np.testing.assert_array_equal(first.gradient, [7, -3])
        np.testing.assert_array_equal(second.gradient, [0, 0])
        np.testing.assert_array_equal(first.value, [1, 2])
        np.testing.assert_array_equal(supplied, [1, 2])
        self.assertFalse(np.shares_memory(first.gradient, first.value))
        self.assertFalse(np.shares_memory(first.gradient, second.gradient))

    def test_directly_constructed_leaf_has_no_parents_or_backward_effect(self):
        tensor = Tensor([1.0, 2.0])
        tensor.gradient[...] = [3.0, -4.0]

        self.assertEqual(tuple(tensor.parents), ())
        self.assertTrue(callable(tensor.operation))
        tensor.operation(tensor)

        np.testing.assert_array_equal(tensor.gradient, [3, -4])
        np.testing.assert_array_equal(tensor.value, [1, 2])

    def test_supplied_gradient_list_is_normalized_to_a_matching_float64_array(self):
        tensor = Tensor([1.0, 2.0], gradient=[0.25, -0.5])

        self.assertIsInstance(tensor.gradient, np.ndarray)
        self.assertEqual(tensor.gradient.dtype, np.dtype(np.float64))
        self.assertEqual(tensor.gradient.shape, tensor.value.shape)
        np.testing.assert_array_equal(tensor.gradient, [0.25, -0.5])

    def test_supplied_scalar_gradient_is_a_zero_dimensional_float64_array(self):
        tensor = Tensor(2.0, gradient=0.5)

        self.assertIsInstance(tensor.gradient, np.ndarray)
        self.assertEqual(tensor.gradient.dtype, np.dtype(np.float64))
        self.assertEqual(tensor.gradient.shape, ())
        self.assertEqual(tensor.gradient.item(), 0.5)

    def test_supplied_integer_gradient_array_is_normalized_to_float64(self):
        supplied = np.array([2, -3], dtype=np.int32)

        tensor = Tensor([1.0, 2.0], gradient=supplied)

        self.assertIsInstance(tensor.gradient, np.ndarray)
        self.assertEqual(tensor.gradient.dtype, np.dtype(np.float64))
        self.assertEqual(tensor.gradient.shape, tensor.value.shape)
        np.testing.assert_array_equal(tensor.gradient, [2, -3])
        self.assertEqual(supplied.dtype, np.dtype(np.int32))

    def test_supplied_gradient_storage_is_independent_of_the_callers_array(self):
        supplied = np.array([0.25, -0.5], dtype=np.float64)

        tensor = Tensor([1.0, 2.0], gradient=supplied)
        supplied[...] = 99.0

        np.testing.assert_array_equal(tensor.gradient, [0.25, -0.5])
        self.assertFalse(np.shares_memory(tensor.gradient, supplied))
