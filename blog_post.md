
Have you ever wondered what's inside the black box of a neural network? It's a
fascinating world of math, code, and data coming together to solve complex
problems. In this post, we'll explore a project that demystifies neural
networks by building one from the ground up in Go. We'll even take a peek at a
TypeScript port!

## What is a Neural Network?

At its core, a neural network is a computational model inspired by the human
brain. It consists of interconnected nodes, or "neurons," organized in layers.
These networks can learn from data and make predictions or decisions.

This project implements a feed-forward neural network, where information flows
in one direction—from the input layer, through one or more hidden layers, to
the output layer.

## The Building Blocks: Neurons and Layers

Think of a neuron as a small computational unit. It receives inputs, processes
them, and produces an output. Each connection between neurons has a "weight,"
which determines the strength of the signal. The neuron also has a "bias,"
which can be thought of as a thumb on the scale, helping to adjust the output.

Here's how the Go project structures the network:

```go
// internal/neuralnetwork/neural_network.go

type NeuralNetwork struct {
  NumInputs             int           `json:"numInputs"`
  HiddenLayers          []int         `json:"hiddenLayers"`
  NumOutputs            int           `json:"numOutputs"`
  HiddenWeights         [][][]float64 `json:"hiddenWeights"`
  OutputWeights         [][]float64   `json:"outputWeights"`
  HiddenBiases          [][]float64   `json:"hiddenBiases"`
  OutputBiases          []float64     `json:"outputBiases"`
  HiddenActivations     []string      `json:"hiddenActivations"`
  OutputActivation      string        `json:"outputActivation"`
 // ...
}
```

This `struct` defines the architecture of our network, including the number of
inputs, the size of the hidden layers, and the number of outputs. It also
stores the weights and biases for each layer.

## The Magic of Math: Forward Propagation

So, how does the network actually "think"? The process of passing input data
through the network to get an output is called **forward propagation**. For each
neuron, we calculate a weighted sum of its inputs, add the bias, and then
apply an "activation function."

The formula looks like this:

$output = f(\sum_{i=1}^{n} (input_i \cdot weight_i) + bias)$

Where `f` is the activation function.

### Activation Functions: The "Spark"

Activation functions are a crucial component. They introduce non-linearity into
the network, allowing it to learn complex patterns. Without them, the network
would just be a simple linear model.

This project supports several activation functions, including:

* **ReLU (Rectified Linear Unit):** A popular choice for hidden layers. It's
    simple and efficient.
    $f(x) = \max(0, x)$
* **Sigmoid:** Often used for binary classification, as it squashes the
    output to a range between 0 and 1.
    $f(x) = \frac{1}{1 + e^{-x}}$

Here's a snippet of the `FeedForward` function in Go:

```go
// internal/neuralnetwork/neural_network.go

func (nn *NeuralNetwork) FeedForward(inputs []float64) ([][]float64, []float64) {
 // ...
 // Calculate hidden layer outputs
 for i, layerSize := range nn.HiddenLayers {
  // ...
  for j := range hiddenOutputs[i] {
   sum := 0.0
   for k, val := range layerInput {
    sum += val * nn.HiddenWeights[i][j][k]
   }
   hiddenOutputs[i][j] = nn.hiddenActivationFuncs[i].Activate(sum + nn.HiddenBiases[i][j])
  }
  layerInput = hiddenOutputs[i]
 }
 // ...
 return hiddenOutputs, finalOutputs
}
```

## Learning from Mistakes: Backpropagation

A neural network learns by adjusting its weights and biases to minimize the
difference between its predictions and the actual target values. This process
is called **training**.

The core of the training process is an algorithm called **backpropagation**. It
works by:

1. Performing a forward pass to get the network's prediction.
1. Calculating the "error" or "loss"—how far off the prediction is from the
    truth.
1. Propagating the error backward through the network, from the output layer
    to the input layer.
1. Using the error to calculate the gradient (the direction of steepest
    ascent) of the loss function with respect to each weight and bias.
1. Updating the weights and biases in the opposite direction of the
    gradient, thus minimizing the error. This step is called **gradient
    descent**.

The `Backpropagate` function in the Go code handles this complex process:

```go
// internal/neuralnetwork/neural_network.go

func (nn *NeuralNetwork) Backpropagate(inputs []float64, targets []float64,
  hiddenOutputs [][]float64, finalOutputs []float64, learningRate float64) {
 // Calculate output layer errors and deltas
 // ...
 // Calculate hidden layer errors and deltas
 // ...
 // Update output weights and biases
 // ...
 // Update hidden weights and biases
 // ...
}
```

## A Tale of Two Languages: The TypeScript Port

One of the interesting aspects of this project is the TypeScript port. It's a
great way to see how the same concepts can be implemented in different
languages. The structure and logic are remarkably similar, showcasing the
universality of neural network principles.

Here's a look at the `NeuralNetwork` class in TypeScript:

```typescript
// ts_port/neural_network.ts

export class NeuralNetwork {
  numInputs: number;
  hiddenLayers: number[];
  numOutputs: number;
  // ...

  constructor(
    inputs: number,
    hiddenLayers: number[],
    outputs: number,
    hiddenActivations: string[],
    outputActivation: string
  ) {
    // ...
  }

  feedForward(inputs: number[]): { hiddenOutputs: number[][], 
    finalOutputs: number[] } {
    // ...
  }

  backpropagate(inputs: number[], targets: number[], hiddenOutputs: number[][],
    finalOutputs: number[], learningRate: number): void {
    // ...
  }

  async train(inputs: number[][], targets: number[][], epochs: number,
    learningRate: number, errorGoal: number, progressCallback?: (error: number) => void): Promise<void> {
    // ...
  }
}
```

## Putting It All Together: The Iris Dataset

The project includes the famous Iris dataset, which is a classic dataset for
classification tasks. The goal is to predict the species of an iris flower
based on its sepal and petal measurements. The TUI makes it easy to train a new
model on this dataset and see the results.

## Conclusion

Building a neural network from scratch is a fantastic way to learn the
fundamentals of deep learning. This Go project provides a clear and concise
implementation that's perfect for digging into the code and understanding how
things work under the hood.

Whether you're a Go developer looking to explore machine learning or a data
scientist curious about Go, this project has something for you. I encourage you
to clone the repository, run the code, and experiment with different network
architectures and datasets. Happy coding!

