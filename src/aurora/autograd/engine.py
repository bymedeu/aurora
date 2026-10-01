from scalar_node import AutoNode


def engine(output: AutoNode):
    L = []
    t = []

    def visit(node: AutoNode):
        if node in t or node in L:
            return
        t.append(node)
        for p in node.parents:
            visit(p)

        L.append(node)
        t.remove(node)

    visit(output)
    return L


def backward(output: AutoNode):
    L = engine(output)
    n: AutoNode
    output.gradient = 1
    for n in reversed(L):
        n.operation(n)
