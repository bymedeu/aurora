def addition(node):
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient
    par1.gradient += node.gradient
    return


def substraction(node):
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient
    par1.gradient += -node.gradient
    return


def division(node):
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += (1 / par1.value) * node.gradient
    par1.gradient += (-par0.value / (par1.value**2)) * node.gradient
    return


def multiplication(node):
    assert len(node.parents) == 2
    par0 = node.parents[0]
    par1 = node.parents[1]
    par0.gradient += node.gradient * par1.value
    par1.gradient += node.gradient * par0.value
    return


def no_op(node):
    return


class AutoNode:
    value = 0
    operation = no_op
    parents: tuple = ()
    gradient = 0

    def __init__(self, value, parents=(), operation=no_op, gradient=0) -> None:
        self.value = value
        self.parents = parents
        self.operation = operation
        self.gradient = gradient
        return

    def __add__(self, other):
        return AutoNode(self.value + other.value, (self, other), addition)

    def __sub__(self, other):
        return AutoNode(self.value - other.value, (self, other), substraction)

    def __mul__(self, other):
        return AutoNode(self.value * other.value, (self, other), multiplication)

    def __truediv__(self, other):
        if other.value == 0:
            raise ZeroDivisionError
        return AutoNode(self.value / other.value, (self, other), division)
