from aurora.autograd import AutoNode


def gradient_descent_step(
    parameters: list[AutoNode], learning_rate: float
) -> list[AutoNode]:
    res = []
    for parameter in parameters:
        res.append(AutoNode(parameter.value - learning_rate * parameter.gradient))
    return res
