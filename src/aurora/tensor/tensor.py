import numpy as np


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


def op_sum(node):
    assert len(node.parents) == 1
    par = node.parents[0]
    par.gradient += node.gradient


def op_none(node):
    return


class Tensor:
    def __init__(self, value, parents=(), operation=op_none, gradient=None) -> None:
        self._value = np.array(value, dtype=np.float64, copy=True)
        self.parents = parents
        self.operation = operation
        if gradient is None:
            self._gradient = np.zeros(self.value.shape)
        else:
            self._gradient = np.array(gradient, dtype=np.float64, copy=True)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @gradient.setter
    def gradient(self, new_grad):
        self._gradient = np.array(new_grad, dtype=np.float64, copy=True)

    def __add__(self, other):
        return Tensor(self.value + other.value, (self, other), op_addition)

    def __sub__(self, other):
        return Tensor(self.value - other.value, (self, other), op_substraction)

    def __mul__(self, other):
        return Tensor(self.value * other.value, (self, other), op_multiplication)

    def __truediv__(self, other):
        if other.value == 0:
            raise ZeroDivisionError
        return Tensor(self.value / other.value, (self, other), op_division)

    def __pow__(self, exp: int):
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
        return Tensor(self.value.sum(), parents=(self,), operation=op_sum)

    def clear_gradient(self):
        self.gradient = np.zeros(self.gradient.shape)
