"""Train the M3 Aurora model using its scalar autograd engine."""

from aurora import Aurora
from aurora.autograd import AutoNode
from aurora.nn import mean_squared_error


def main() -> None:
    inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    targets = [-3.0, -1.0, 1.0, 3.0, 5.0]
    model = Aurora()

    print("Training pairs:", list(zip(inputs, targets)))
    print(f"Before: weight={model.weight.value:.6f}, bias={model.bias.value:.6f}")
    print(f"Before: predict(4)={model.predict(4.0).value:.6f}")

    # fit builds prediction/loss graphs, calls backward, and retains new parameters.
    model.fit(inputs, targets, steps=50, learning_rate=0.1)

    print(f"After:  weight={model.weight.value:.6f}, bias={model.bias.value:.6f}")
    predictions = [model.predict(x) for x in inputs]
    target_nodes = [AutoNode(y) for y in targets]
    loss = mean_squared_error(predictions, target_nodes)
    print(f"Final training MSE: {loss.value:.6e}")

    print("Predictions for inputs outside the training set:")
    for value in (-3.0, 0.5, 4.0):
        prediction = model.predict(value)
        print(f"  x={value:5.1f} -> {prediction.value:.6f}")


if __name__ == "__main__":
    main()
