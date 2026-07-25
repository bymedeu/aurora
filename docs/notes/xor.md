# XOR - 2026-06-29

# Small Neural Network

## 1. From one neuron to a layer

Now that we have seen what happens with one neuron, we must link them all together. We now have multiple vectors of neurons. The first vector is called the `input layer`, that's where the data is entered. The last vector is called the `output layer`, that is where we get the result. Between them, there may be up to n vectors of neurons, called `hidden layers`. Hidden layers add complexity and more possible operations on an input to a result.

## 2. Shapes of inputs, weights, biases and outputs

Since there are a list of neurons for each layer, having one object per neuron would be highly unoptimised. The inputs/outputs are then being used as matrixes. Each layer has it's matrix of weights for each neuron. We take the vector of outputs of the previous layer, multiply it by the matrix of each weight for each neuron, and then we add the result with the matrix of biaises. Let np be the number of neurons on the previous layer, nc the number of neurons on the current layer, the sises of each matrixes are: [np, 1], [np, nc], [1, nc]. Thus, the resultig matrix will be of size [1, nc]. Obviousely, it can be transposed to become [nc, 1].

## 3. Forward propagation

The formula for forward propagation is:
Let x be the matrix of inputs in the layer, w the matrix of weights, b the matrix of biaises

$$ x \cdot w + b $$

Then, for each element of the resulting matrix, we apply the activation function (sigmoid in our case), and we get the result.

## 4. Cost and gradient descent

Basically the same thing as forward propagation, with matrix multiplications, but with the formulas unsed in @file single_neuron.md

## Sources

- [Yes, you shoud understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
- [A step by step backpropagation example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
- [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)

