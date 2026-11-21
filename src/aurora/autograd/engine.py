import numpy as np

from aurora.tensor.tensor import Tensor

from .scalar_node import AutoNode


def _topological_order(output: AutoNode):
    order = []
    visiting = set()
    visited = set()

    def visit(node: AutoNode):
        if node in visited:
            return
        if node in visiting:
            raise ValueError("Autograd graph contains a cycle")
        visiting.add(node)
        for p in node.parents:
            visit(p)
        visiting.remove(node)
        visited.add(node)
        order.append(node)

    visit(output)
    return order


def backward(output: AutoNode):
    L = _topological_order(output)
    n: AutoNode
    output.gradient = 1
    for n in reversed(L):
        n.operation(n)


def clear_gradients(output: AutoNode):
    L = _topological_order(output)
    for n in reversed(L):
        if type(n) is AutoNode:
            n.gradient = 0
        if type(n) is Tensor:
            n.gradient = np.zeros(n.gradient.shape)
