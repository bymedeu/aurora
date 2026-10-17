from aurora.autograd import AutoNode, backward
from aurora.nn import mean_squared_error
from aurora.optim.gradient_descent import gradient_descent_step


class Aurora:
    def __init__(self, weight: float = 0.0, bias: float = 0.0) -> None:
        self.weight = AutoNode(weight)
        self.bias = AutoNode(bias)

    def predict(self, value: float = 0.0) -> AutoNode:
        return self.weight * AutoNode(value) + self.bias

    def fit(
        self,
        inputs: list[float],
        targets: list[float],
        steps: int = 50,
        learning_rate: float = 0.1,
    ) -> None:
        for s in range(steps):
            preds = []
            trgts = [AutoNode(t) for t in targets]
            for i in inputs:
                preds.append(self.predict(i))
            loss = mean_squared_error(preds, trgts)
            backward(loss)
            [self.weight, self.bias] = gradient_descent_step(
                [self.weight, self.bias], learning_rate
            )
