# Single neuron - 2026-06-14

Objectives
1. Study the concept of an activation function and binary classification.
2. Answer these guiding questions in your note:
    • What happens mathematically when we feed the linear output ax + b into a Sigmoid function
    • What does the output represent?
    • Geometrically, what does a decision boundary look like for a single neuron?


## What is a neuron

In short, a neuron is a number, raising from 0 to 1. The number is called `activation`. When given values from other neurons, it will apply different weights for each neurons, and output a new value. For example, when feeding it the `x` value, it will apply $w * x + b$ and output a value `z`. The value is then passed in a `sigmoid function`, that will return a value between `0` and `1`, which we call `y`. 
The output `y` is then the probability of `x` meeting the requirements needed for the neuron. 

> [!IMPORTANT] Sigmoid function
> A sigmoid function is a function that takes a value `x` and that returns an float between `0` and `1`. The most common sigmoid is:
> $$\frac{1}{\left(1 + e^{-x})}$$
> This range is really useful since it simplifies a lot everyting, the maximum value is now `1`, and the minimum is now `0`.

## Geometrically

If we use 2 groups of blobs, and use a neural network to differenciate them, the border between those 2 groups will be when the neuron gives out the `0.5` value. Following the $w * x + b$ formula, this will then be a line. Looking back at the sigmoid function, `y` will be equal to `0.5` only when `z` is equal to `0`. This marks the indecisiveness of the neuron.
When using multiple neurons, the result will always be linear, whether it is a point (1 input), a line (2 inputs), a plane (3 inputs), or more.
For better classification, we use the `Binary Cross Entropy (BCE)` formula to help us determine whether a value between `0` and `1` should be in a group or not. BCE measures the distance between the true labels and the predicted probabilities. The closer `p_i` is to `y_i`, the lower the value is, the better it is. We calculate each loss, and get the average of all the losses. The closer the value is to 0, the better the neuron is.

> [!IMPORTANT] Binary Cross Entropy (BCE)
> $$ - \frac{1}{N}\sum_{i=1}^{N}\left(y_i\log{p_i} + (1 - y_i)\log{1-p_i}\right) $$
> Where `N` is the number of observations, `y_i` is the binary label of the `i`th observation and `p_i` the predicted probability of the `i`th observation being in the class 1 (opposed to class 0)
 

## Training summary

The script in [train.py](../../examples/single_neuron/train.py) is the testing part of this file. I genereated 2 blobs of a variable size and of variable positions, and then created a neuron that differenciated whether a point was a part of Blob 0 or Blob 1. It first starts by using random weights, and using gradient loss, updates the weights to better group points. The neuron was run over 400 epochs, and printed each time the BCE. It was going smaller and smaller, which meant that the weights were adapting. As an example, one run got (w1=1.175, w2=1.197, b=-0.026) as weights / bias, with a BCE of 0.0915 after 400 epochs. These values trace a line used to separate both groups of points. 
