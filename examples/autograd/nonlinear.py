from aurora.autograd import AutoNode, backward
from aurora.nn import mean_squared_error
from aurora.optim import gradient_descent_step

"""

in the list, here are the weights by index:
0 - w1
1 - b1
2 - w2
3 - b2 
4 - v1
5 - v2
6 - c

"""


def forward(input: AutoNode, parameter: list[AutoNode]):
    h1 = (parameter[0] * input + parameter[1]).sigmoid()
    h2 = (parameter[2] * input + parameter[3]).sigmoid()
    y = parameter[4] * h1 + parameter[5] * h2 + parameter[6]
    return y


if __name__ == "__main__":
    parameters = [AutoNode(t) for t in [1, 0, -1, 0, 1, -1, 0]]
    inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    targets = [0.1, 0.2, 0.5, 0.8, 0.9]
    trgts = [AutoNode(t) for t in targets]
    preds = []
    learning_rate = 0.1
    steps = 50

    for s in range(steps):
        preds = []
        for i in inputs:
            preds.append(forward(AutoNode(i), parameters))
        loss = mean_squared_error(preds, trgts)
        backward(loss)
        parameters = gradient_descent_step(parameters, learning_rate)
