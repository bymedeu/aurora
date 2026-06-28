# Linear regression - 2026-06-07:2026-06-10

Objectives:
1. What does the model  y = ax + b  represent geometrically?
2. What does it mean for a prediction to be "wrong"? How do you measure that?
3. If you had to improve the model, which direction would you move  a  and  b ? Why?

Linear regression is the most basic method used to train a model. You give it an input `x`, and it will give you an output `y` according to the formula cited below.

## y = ax + b

The formula used for linear regression is $y=a\cdot x+b$. It uses 4 parameters:
- `x`: the input data
- `a` (or `w`): the weight of the data. It determies how much the prediction changes when `x` changes
- `b`: the bias, the prediction when $x=0$
- `y`: the output of the formula

With this formula, an AI will be able to estimate an answer to a basic question, given an input data `x`.
Visually, this will represent a line that approximates values (shown as dots) on a graph, where the line will be as close as possible to every point. `b` is the value that the line will hold at $x=0$, and `a` is the factor value, each time `x` increases by 1, `y` increases by `a`.


## How to determine these values

`a` and `b` are values that must be appropriately determined for the data to be effective. The model needs to know whether `a` and `b` are good. To do this, the model uses a method called a **loss function**.
The most common method used is the **mean squared method**. The method calculates how wrong the current `a` and `b` values are. The higher the `loss` value is, the more wrong it is. We start by initialising `a` and `b` (most of the time at 0).

> [!IMPORTANT] Mean Squared Formula
> $$\text loss = \frac{1}{n} \cdot \left(\sum{(y_model - y_expected)²)}\right$$

Then, by using this method, the model will then use the **gradient descent** method. Once that we have our `loss` data, we then use the gradient descent formula.


> [!IMPORTANT] Gradient Descent Forumula
> For the slope `a`:
> $$ \frac{\partial J}{\partial a}  = -\frac{2}{n}\sum_{i=1}^{n} x_i \left(y_i - (a x_i + b)\right) $$
> 
> For the bias `b`:
> $$ \frac{\partial J}{\partial b}  = -\frac{2}{n}\sum_{i=1}^{n} \left(y_i - (a x_i + b)\right) $$

Once the gradient calculated for `a` and `b`, we can now calculate their new values.

> [!IMPORTANT] New values for `a` and `b`
> For the slope `a`:
> $$a=a-\alpha \cdot \frac{\partial J}{\partial a}$$
> 
> For the bias `b`:
> $$ b=b-\alpha \cdot \frac{\partial J}{\partial b}$$

We then repeat the gradient loss method until the variation of the mean squared gets very low, we have reached the most adapted value.

## Sources

- https://developers.google.com/machine-learning/crash-course/linear-regression
- https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/
- https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- https://www.geeksforgeeks.org/machine-learning/gradient-descent-algorithm-and-its-variants/
