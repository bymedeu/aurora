"""Traverse computation graphs and coordinate their local backward rules."""

import numpy as np

from aurora.tensor import Tensor

from .scalar_node import AutoNode


def _topological_order(output: AutoNode | Tensor):
    """Return reachable nodes with each parent preceding its dependent nodes.

    Args:
        output: Scalar or tensor node at the end of the graph.

    Returns:
        A list containing each reachable node once, with output last.

    Raises:
        ValueError: A cycle is found in the parent relationships.

    This traversal does not execute operations or change gradients.
    """
    order = []
    visiting = set()
    visited = set()

    def visit(node: AutoNode | Tensor):
        """Visit parents before recording this node, rejecting graph cycles."""
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
    """Seed the output gradient and propagate contributions through its graph.

    Args:
        output: An exact AutoNode or Tensor instance with a valid recorded graph.

    An AutoNode output receives scalar seed 1. A Tensor output receives an array
    of ones with its current gradient shape, which must match its value shape.
    For a non-scalar tensor, that seed differentiates the sum of its elements;
    it does not construct a full Jacobian. Each recorded operation must support
    the shapes used in the graph.

    Other gradients are not cleared: local rules accumulate into their parents.
    Clear the graph before repeating backward on the same intermediates.

    Raises:
        ValueError: A cycle is found before any gradient is seeded.
    """
    L = _topological_order(output)
    if type(output) == AutoNode:
        output.gradient = 1
    elif type(output) == Tensor:
        output.gradient = np.ones(output.gradient.shape)
    n: AutoNode | Tensor
    for n in reversed(L):
        n.operation(n)


def clear_gradients(output: AutoNode | Tensor):
    """Ask every reachable node to reset its own gradient.

    Args:
        output: Scalar or tensor node whose ancestor gradients should be cleared.

    Values, parent relationships, and backward operations remain unchanged.
    Nodes outside this graph are untouched. Each node owns the numerical reset
    behavior; Tensor resets preserve the current gradient array shape.

    Raises:
        ValueError: A cycle is found before any gradients are cleared.
    """
    L = _topological_order(output)
    for n in reversed(L):
        n.clear_gradient()
