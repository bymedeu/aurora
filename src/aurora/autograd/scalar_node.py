"""Scalar graph nodes and local derivative rules used by Aurora autograd."""

from math import exp


def op_addition(node):
    """Accumulate the incoming gradient into both addition parents."""
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient
    par1.gradient += node.gradient


def op_substraction(node):
    """Accumulate the incoming gradient with opposite signs into the parents."""
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient
    par1.gradient += -node.gradient


def op_division(node):
    """Accumulate the quotient-rule contributions into both scalar parents.

    The denominator must be nonzero; contributions use the saved parent values.
    """
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += (1 / par1.value) * node.gradient
    par1.gradient += (-par0.value / (par1.value**2)) * node.gradient


def op_multiplication(node):
    """Accumulate each product derivative using the other parent value.

    Repeated parents receive both contributions through the two parent positions.
    """
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient * par1.value
    par1.gradient += node.gradient * par0.value


def op_sigmoid(node):
    """Accumulate the incoming gradient times output * (1 - output)."""
    assert len(node.parents) == 1
    par = node.parents[0]
    par.gradient += node.gradient * node.value * (1 - node.value)


def op_none(node):
    """Leave gradients unchanged for a leaf with no recorded operation."""
    return


class AutoNode:
    """Represent a scalar value and one node in a computation graph.

    Arithmetic creates connected result nodes. The shared engine visits the graph
    and calls each stored operation to accumulate gradients into its parents.

    Attributes:

    - `value`: Numerical scalar exposed through a read-only property.
    - `gradient`: Mutable accumulated derivative of the differentiated
      objective.
    - `parents`: Immediate input nodes, retaining repeated operand positions.
    - `operation`: Local backward callback accepting this result node and
      returning None; leaf nodes use a no-op callback.
    """

    def __init__(self, value, parents=(), operation=op_none, gradient=0) -> None:
        """Store the scalar state and the operation that produced it.

        Parameters:

        - `value`: Numerical scalar represented by the node.
        - `parents`: Immediate dependencies; empty for a directly constructed
          leaf.
        - `operation`: Local backward callback accepting the result node.
        - `gradient`: Initial accumulated gradient, defaulting to zero.
        """
        self._value = value
        self.parents = parents
        self.operation = operation
        self.gradient = gradient

    @property
    def value(self):
        """The stored scalar value; this property has no assignment setter."""
        return self._value

    def __add__(self, other):
        """Return a connected scalar sum of this node and another AutoNode."""
        return AutoNode(self.value + other.value, (self, other), op_addition)

    def __sub__(self, other):
        """Return a connected scalar difference from another AutoNode."""
        return AutoNode(self.value - other.value, (self, other), op_substraction)

    def __mul__(self, other):
        """Return a connected scalar product with another AutoNode."""
        return AutoNode(self.value * other.value, (self, other), op_multiplication)

    def __truediv__(self, other):
        """Return a connected scalar quotient.

        Parameters:

        - `other`: Denominator AutoNode.

        Raises:

        - `ZeroDivisionError`: The denominator value is zero.
        """
        if other.value == 0:
            raise ZeroDivisionError
        return AutoNode(self.value / other.value, (self, other), op_division)

    def __pow__(self, exp: int):
        """Build a non-negative integer power from connected multiplications.

        Parameters:

        - `exp`: Integer exponent. Zero returns a constant leaf containing one.

        Raises:

        - `TypeError`: The exponent is not an integer.
        - `ValueError`: The exponent is negative.
        """
        if not isinstance(exp, int):
            raise TypeError("Exponent must be an integer")
        if exp < 0:
            raise ValueError("Exponent must be non-negative")
        par = AutoNode(1)
        b = self
        while exp > 0:
            if exp & 1:
                par = par * b
            b = b * b
            exp >>= 1
        return par

    def sigmoid(self):
        """Return a connected sigmoid result using a stable scalar evaluation."""
        if self.value >= 0:
            value = 1.0 / (1.0 + exp(-self.value))
        else:
            exponential = exp(self.value)
            value = exponential / (1.0 + exponential)

        return AutoNode(value, (self,), op_sigmoid)

    def clear_gradient(self):
        """Reset this node gradient to scalar zero without traversing parents."""
        self.gradient = 0
