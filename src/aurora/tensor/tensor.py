"""NumPy-backed graph nodes and their local array derivative rules."""

import numpy as np


def op_addition(node):
    """Accumulate the incoming array gradient into both addition parents.

    The current backward contract covers equal-shape operands; NumPy forward
    broadcasting alone does not provide the required broadcast reduction rule.
    """
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient
    par1.gradient += node.gradient


def op_substraction(node):
    """Accumulate signed incoming gradients into the two subtraction parents."""
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient
    par1.gradient += -node.gradient


def op_division(node):
    """Accumulate quotient derivatives using the saved parent values."""
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += (1 / par1.value) * node.gradient
    par1.gradient += (-par0.value / (par1.value**2)) * node.gradient


def op_multiplication(node):
    """Accumulate elementwise product derivatives into both parents.

    Each contribution multiplies the incoming gradient by the other parent value.
    Repeated operands retain both contributions. Broadcasting derivatives still
    require separate shape-reduction support.
    """
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient * par1.value
    par1.gradient += node.gradient * par0.value


def op_sum(node):
    """Accumulate a full-array sum result gradient into every parent element.

    The result has one parent and a scalar incoming gradient. NumPy broadcasts
    that contribution across the parent gradient array without changing its shape.
    """
    assert len(node.parents) == 1
    par = node.parents[0]
    par.gradient += node.gradient


def op_none(node):
    """Leave gradients unchanged for a tensor leaf without an operation."""
    return


class Tensor:
    """Store an owned float64 array together with its autograd graph state.

    Scalar inputs become zero-dimensional arrays. Arithmetic records parents and
    local backward callbacks; the shared autograd engine performs traversal.
    NumPy storage does not itself record Aurora graph dependencies.

    Attributes:
        value: Owned array. The property has no assignment setter, but its contents
            remain writable; this is not a general immutability guarantee.
        gradient: Independently stored, writable float64 array. A valid gradient
            matches the value shape; assignments do not yet validate that invariant.
        parents: Immediate input nodes, empty for a directly constructed leaf.
        operation: Local backward callback accepting this result node and returning
            None. Leaf tensors use a no-op callback.

    Forward broadcasting accepted by NumPy is not general backward support.
    """

    def __init__(self, value, parents=(), operation=op_none, gradient=None) -> None:
        """Copy numerical inputs into owned float64 array storage.

        Args:
            value: Numerical scalar, sequence, or array, preserving its input shape.
            parents: Immediate dependencies; empty for a directly constructed leaf.
            operation: Local backward callback accepting this result tensor.
            gradient: Initial gradient data copied independently. None creates zeros
                matching the value shape. Supplied gradient shapes are not validated.
        """
        self._value = np.array(value, dtype=np.float64, copy=True)
        self.parents = parents
        self.operation = operation
        if gradient is None:
            self._gradient = np.zeros(self.value.shape)
        else:
            self._gradient = np.array(gradient, dtype=np.float64, copy=True)

    @property
    def value(self):
        """The owned numerical array, whose contents remain writable."""
        return self._value

    @property
    def gradient(self):
        """The independently stored array of accumulated gradient contributions."""
        return self._gradient

    @gradient.setter
    def gradient(self, new_grad):
        """Copy a supplied gradient into independent float64 storage.

        Assignment preserves the supplied shape; it does not validate it against value
        or calculate derivatives. The backward rules supply derivative contributions.
        """
        self._gradient = np.array(new_grad, dtype=np.float64, copy=True)

    def __add__(self, other):
        """Return a connected NumPy sum of this tensor and another Tensor.

        The verified local backward rule covers operands of equal shape.
        """
        return Tensor(self.value + other.value, (self, other), op_addition)

    def __sub__(self, other):
        """Return a connected elementwise difference from another Tensor."""
        return Tensor(self.value - other.value, (self, other), op_substraction)

    def __mul__(self, other):
        """Return a connected elementwise product with another Tensor."""
        return Tensor(self.value * other.value, (self, other), op_multiplication)

    def __truediv__(self, other):
        """Return a connected quotient using the current scalar-oriented zero guard.

        Args:
            other: Denominator tensor.

        Raises:
            ZeroDivisionError: The scalar zero guard detects a zero denominator.

        Validation of general array denominators is not implemented; multi-element
        arrays can raise an ambiguous-truth-value error before the division.
        """
        if other.value == 0:
            raise ZeroDivisionError
        return Tensor(self.value / other.value, (self, other), op_division)

    def __pow__(self, exp: int):
        """Build a non-negative integer power from connected multiplications.

        Args:
            exp: Integer exponent. Zero returns a constant scalar tensor containing one.

        Raises:
            TypeError: The exponent is not an integer.
            ValueError: The exponent is negative.
        """
        if not isinstance(exp, int):
            raise TypeError("Exponent must be an integer")
        if exp < 0:
            raise ValueError("Exponent must be non-negative")
        par = Tensor(1)
        b = self
        while exp > 0:
            if exp & 1:
                par = par * b
            b = b * b
            exp >>= 1
        return par

    def sum(self):
        """Sum all array elements while preserving the graph connection.

        Returns:
            A scalar Tensor with shape (), this tensor as its sole parent, and the
            full-array sum backward rule. Axis selection is not supported.

        Backward accumulates the scalar incoming gradient into every parent element.
        The forward calculation leaves input values and existing gradients unchanged.
        """
        return Tensor(self.value.sum(), parents=(self,), operation=op_sum)

    def clear_gradient(self):
        """Reset this tensor gradient to an independently stored zero array.

        The current gradient shape is preserved; a valid gradient already matches the
        value shape. Values and graph metadata are unchanged, and parents are not
        traversed by this method.
        """
        self.gradient = np.zeros(self.gradient.shape)
