import numpy as np

from aurora.tensor import Tensor

from .scalar_node import AutoNode


def _topological_order(output: AutoNode | Tensor):
    order = []
    visiting = set()
    visited = set()

    def visit(node: AutoNode | Tensor):
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


def backward(output: AutoNode | Tensor):
    L = _topological_order(output)
    if type(output) == AutoNode:
        output.gradient = 1
    elif type(output) == Tensor:
        output.gradient = np.ones(output.gradient.shape)
    n: (
        AutoNode | Tensor
    )  # why can't I force it to one of both, and it says unknown as type iun the lsp? Why can't I see the description of operation even with this precision? Hoiw to fix that ?
    for n in reversed(L):
        n.operation(n)


def clear_gradients(output: AutoNode | Tensor):
    L = _topological_order(output)
    for n in reversed(L):
        n.clear_gradient()
