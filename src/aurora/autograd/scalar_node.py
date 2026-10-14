from math import exp


def op_addition(node):
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient
    par1.gradient += node.gradient


def op_substraction(node):
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient
    par1.gradient += -node.gradient


def op_division(node):
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += (1 / par1.value) * node.gradient
    par1.gradient += (-par0.value / (par1.value**2)) * node.gradient


def op_multiplication(node):
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient * par1.value
    par1.gradient += node.gradient * par0.value


def op_sigmoid(node):
    assert len(node.parents) == 1
    par = node.parents[0]
    par.gradient += node.gradient * node.value * (1 - node.value)


def op_none(node):
    return


class AutoNode:
    def __init__(self, value, parents=(), operation=op_none, gradient=0) -> None:
        self._value = value
        self.parents = parents
        self.operation = operation
        self.gradient = gradient

    @property
    def value(self):
        return self._value

    def __add__(self, other):
        return AutoNode(self.value + other.value, (self, other), op_addition)

    def __sub__(self, other):
        return AutoNode(self.value - other.value, (self, other), op_substraction)

    def __mul__(self, other):
        return AutoNode(self.value * other.value, (self, other), op_multiplication)

    def __truediv__(self, other):
        if other.value == 0:
            raise ZeroDivisionError
        return AutoNode(self.value / other.value, (self, other), op_division)

    def __pow__(self, exp: int):
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
        if self.value >= 0:
            value = 1.0 / (1.0 + exp(-self.value))
        else:
            exponential = exp(self.value)
            value = exponential / (1.0 + exponential)

        return AutoNode(value, (self,), op_sigmoid)
