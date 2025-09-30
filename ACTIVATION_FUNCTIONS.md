# Activation Functions

This document provides an overview of the activation functions available in
this neural network implementation. Each activation function is responsible for
introducing non-linearity into the network, allowing it to learn more complex
patterns.

## ReLU (Rectified Linear Unit)

The Rectified Linear Unit (ReLU) is one of the most widely used activation
functions in deep learning, especially in hidden layers. It is computationally
efficient and helps to mitigate the vanishing gradient problem that can occur
with other activation functions like sigmoid and tanh.

**Formula:**
$f(x) = \max(0, x)$
This means that if the input `x` is positive, the function returns `x`, and if
it's negative, it returns `0`.

**Derivative:**
The derivative of ReLU is:
$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$
It is undefined at `x = 0`, though it's often set to `0` in practice.

**Use Cases:**

- **Hidden Layers:** It is the default choice for hidden layers in most
feed-forward neural networks.
- **Not for Output Layers:** It is generally not used in the output layer,
especially for regression tasks where negative values might be expected.

## Sigmoid

The Sigmoid function maps any input value to a value between 0 and 1. This
makes it particularly useful for models where we need to predict a probability
as an output.

**Formula:**
$f(x) = \frac{1}{1 + e^{-x}}$

**Derivative:**
$f'(x) = f(x)(1 - f(x))$

**Use Cases:**

- **Binary Classification:** It is often used in the output layer of a binary
classification network, where the output can be interpreted as the probability
of the positive class.
- **Hidden Layers:** While it can be used in hidden layers, it has fallen out
of favor due to the vanishing gradient problem, where the gradients can
become very small, making it difficult for the network to learn.

## Tanh (Hyperbolic Tangent)

The Hyperbolic Tangent (Tanh) function is similar to the sigmoid function, but
it maps input values to a range between -1 and 1. This can be advantageous as
it centers the output around zero, which can sometimes help with convergence
during training.

**Formula:**
$f(x) = \tanh(x)$

**Derivative:**
$f'(x) = 1 - \tanh^2(x)$

**Use Cases:**

- **Hidden Layers:** It is a popular choice for hidden layers, especially in
recurrent neural networks (RNNs).
- **Output Layers:** It can be used in the output layer for regression tasks
where the output is expected to be in the range of -1 to 1.

## Linear

The Linear activation function, as the name suggests, is a linear function that
simply returns the input value. It does not introduce any non-linearity into
the network.

**Formula:**
$f(x) = x$

**Derivative:**
$f'(x) = 1$

**Use Cases:**

- **Output Layer for Regression:** It is the standard choice for the output
layer in regression tasks, where the output can be any real number.
- **Not for Hidden Layers:** Using a linear activation function in a hidden
layer would make the entire network equivalent to a single-layer network,
defeating the purpose of having multiple layers.
