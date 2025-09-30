# Go Neural Network Porting Guide

This document provides a comprehensive guide for programmers looking to port
the Go Neural Network program to other languages. It details the architecture,
core algorithms with pseudocode, and implementation notes.

## Project Overview

The Go Neural Network is a command-line application that allows users to train,
evaluate, and use simple neural networks for both regression and classification
tasks. It features an interactive terminal user interface (TUI) for ease of use.

## High-Level Architecture

The application is built on three main components:

1. **Terminal User Interface (TUI):** The user-facing part of the application.
It's responsible for gathering input from the user and displaying results. This
is the most language- and framework-dependent part of the application.
1. **Data Handling:** This component is responsible for loading CSV files,
automatically detecting the problem type (regression vs. classification),
normalizing data, and splitting it into training and test sets. It also handles
the serialization (saving/loading) of trained models.
1. **Neural Network:** The core of the application. This component defines the
neural network structure and implements the training (feedforward,
backpropagation) and prediction algorithms.

The main application entry point initializes and runs the TUI. The TUI then
calls the data handling and neural network components based on user actions.

---

## Part 1: The Neural Network Core

This is the most critical part of the port. The logic here is language-agnostic.

### Data Structures

You will need a primary data structure to hold the neural network's state.

```
Structure NeuralNetwork:
    numInputs: Integer
    hiddenLayers: Array of Integers (e.g., [20, 20])
    numOutputs: Integer
    hiddenWeights: 3D Array of Floats
    outputWeights: 2D Array of Floats
    hiddenBiases: 2D Array of Floats
    outputBiases: 1D Array of Floats
    // Store activation function names for serialization
    hiddenActivations: Array of Strings
    outputActivation: String
```

### 1. Activation Functions

You need to implement a set of activation functions and their derivatives.

| Function | `Activate(x)` Formula | `Derivative(output)` Formula |
| :--- | :--- | :--- |
| **Sigmoid** | `1 / (1 + exp(-x))` | `output * (1 - output)` |
| **ReLU** | `max(0, x)` | `1 if x > 0 else 0` |
| **Tanh** | `tanh(x)` | `1 - output^2` |
| **Linear** | `x` | `1` |

### 2. Network Initialization

The network's weights and biases must be initialized with small random values.
This implementation uses **He initialization**, which is a good practice for
networks with ReLU activation functions.

**Pseudocode for `InitNetwork`:**

```
FUNCTION InitNetwork(inputs, hiddenLayers, outputs, hiddenActivations, outputActivation):
    network = new NeuralNetwork(...)

    // --- Initialize Hidden Layers ---
    previousLayerSize = inputs
    FOR i FROM 0 TO length(hiddenLayers) - 1:
        layerSize = hiddenLayers[i]
        network.hiddenWeights[i] = new 2D_Array(layerSize, previousLayerSize)
        network.hiddenBiases[i] = new 1D_Array(layerSize) // Initialize to zeros

        // He initialization for weights
        he_init_std_dev = sqrt(2.0 / previousLayerSize)
        FOR j FROM 0 TO layerSize - 1:
            FOR k FROM 0 TO previousLayerSize - 1:
                network.hiddenWeights[i][j][k] = random_normal() * he_init_std_dev

        previousLayerSize = layerSize

    // --- Initialize Output Layer ---
    network.outputWeights = new 2D_Array(outputs, previousLayerSize)
    network.outputBiases = new 1D_Array(outputs) // Initialize to zeros

    he_init_std_dev = sqrt(2.0 / previousLayerSize)
    FOR i FROM 0 TO outputs - 1:
        FOR j FROM 0 TO previousLayerSize - 1:
            network.outputWeights[i][j] = random_normal() * he_init_std_dev

    RETURN network
```

### 3. Feedforward Algorithm

This algorithm computes the network's output for a given input.

**Pseudocode for `FeedForward`:**

```
FUNCTION FeedForward(network, inputs):
    allHiddenOutputs = new 2D_Array()
    currentLayerInput = inputs

    // --- Process Hidden Layers ---
    FOR i FROM 0 TO length(network.hiddenLayers) - 1:
        layerOutput = new 1D_Array(network.hiddenLayers[i])
        FOR j FROM 0 TO length(layerOutput) - 1:
            // Calculate weighted sum
            sum = network.hiddenBiases[i][j]
            FOR k FROM 0 TO length(currentLayerInput) - 1:
                sum = sum + currentLayerInput[k] * network.hiddenWeights[i][j][k]

            // Apply activation function
            activation_func = GetActivation(network.hiddenActivations[i])
            layerOutput[j] = activation_func.Activate(sum)

        add layerOutput to allHiddenOutputs
        currentLayerInput = layerOutput

    // --- Process Output Layer ---
    finalOutputs = new 1D_Array(network.numOutputs)
    FOR i FROM 0 TO length(finalOutputs) - 1:
        // Calculate weighted sum
        sum = network.outputBiases[i]
        FOR j FROM 0 TO length(currentLayerInput) - 1:
            sum = sum + currentLayerInput[j] * network.outputWeights[i][j]

        // Apply activation function
        activation_func = GetActivation(network.outputActivation)
        finalOutputs[i] = activation_func.Activate(sum)

    RETURN allHiddenOutputs, finalOutputs
```

### 4. Backpropagation Algorithm

This is the most complex part. It calculates the error of the network and
updates the weights and biases.

**Pseudocode for `Backpropagate`:**

```
FUNCTION Backpropagate(network, inputs, targets, hiddenOutputs, finalOutputs, learningRate):
    // --- 1. Calculate Output Layer Error ---
    outputErrors = new 1D_Array(network.numOutputs)
    outputDeltas = new 1D_Array(network.numOutputs)
    output_activation_func = GetActivation(network.outputActivation)
    FOR i FROM 0 TO network.numOutputs - 1:
        outputErrors[i] = targets[i] - finalOutputs[i]
        derivative = output_activation_func.Derivative(finalOutputs[i])
        outputDeltas[i] = outputErrors[i] * derivative

    // --- 2. Calculate Hidden Layer Errors (in reverse) ---
    hiddenDeltas = new 2D_Array() // Store deltas for each hidden layer
    nextLayerDeltas = outputDeltas
    nextLayerWeights = network.outputWeights

    FOR i FROM length(network.hiddenLayers) - 1 DOWNTO 0:
        layerDeltas = new 1D_Array(network.hiddenLayers[i])
        hidden_activation_func = GetActivation(network.hiddenActivations[i])
        FOR j FROM 0 TO network.hiddenLayers[i] - 1:
            // Calculate error sum from the next layer
            errorSum = 0.0
            FOR k FROM 0 TO length(nextLayerDeltas) - 1:
                errorSum = errorSum + nextLayerDeltas[k] * nextLayerWeights[k][j]

            derivative = hidden_activation_func.Derivative(hiddenOutputs[i][j])
            layerDeltas[j] = errorSum * derivative
        
        insert layerDeltas at the beginning of hiddenDeltas

        // Prepare for the next iteration (moving backwards)
        nextLayerDeltas = layerDeltas
        nextLayerWeights = network.hiddenWeights[i]

    // --- 3. Update Weights and Biases ---

    // Update output layer
    lastHiddenLayerOutput = hiddenOutputs[length(hiddenOutputs) - 1]
    FOR i FROM 0 TO network.numOutputs - 1:
        FOR j FROM 0 TO length(lastHiddenLayerOutput) - 1:
            network.outputWeights[i][j] += learningRate * outputDeltas[i] * lastHiddenLayerOutput[j]
        network.outputBiases[i] += learningRate * outputDeltas[i]

    // Update hidden layers
    FOR i FROM length(network.hiddenLayers) - 1 DOWNTO 0:
        previousLayerOutput = (i == 0) ? inputs : hiddenOutputs[i-1]
        FOR j FROM 0 TO network.hiddenLayers[i] - 1:
            FOR k FROM 0 TO length(previousLayerOutput) - 1:
                network.hiddenWeights[i][j][k] += learningRate *
hiddenDeltas[i][j] * previousLayerOutput[k]
            network.hiddenBiases[i][j] += learningRate * hiddenDeltas[i][j]
```

### 5. Training Loop

The training process iterates over the dataset for a number of "epochs",
running feedforward and backpropagation for each data sample.

**Pseudocode for `Train`:**

```
FUNCTION Train(network, trainingInputs, trainingTargets, epochs, learningRate, errorGoal):
    FOR e FROM 0 TO epochs - 1:
        totalError = 0.0
        FOR i FROM 0 TO length(trainingInputs) - 1:
            // Get a single training sample
            inputSample = trainingInputs[i]
            targetSample = trainingTargets[i]

            // 1. Feedforward
            hiddenOutputs, finalOutputs = FeedForward(network, inputSample)

            // 2. Backpropagate
            Backpropagate(network, inputSample, targetSample, hiddenOutputs,
finalOutputs, learningRate)

            // 3. Accumulate error (e.g., Mean Squared Error)
            FOR j FROM 0 TO length(targetSample) - 1:
                totalError += 0.5 * (targetSample[j] - finalOutputs[j])^2

        // Check for early stopping
        averageError = totalError / length(trainingInputs)
        IF averageError < errorGoal:
            BREAK // Stop training

        // Optional: Report progress (epoch number, average error)
```

---

## Part 2: Data Handling

### Data Structures

You'll need a structure to hold the loaded dataset and another to hold the
complete model for serialization.

```
Structure Dataset:
    TrainInputs: 2D Array of Floats
    TrainTargets: 2D Array of Floats
    TestInputs: 2D Array of Floats
    TestTargets: 2D Array of Floats
    InputSize: Integer
    OutputSize: Integer
    // For denormalization
    InputMins: 1D Array of Floats
    InputMaxs: 1D Array of Floats
    TargetMins: 1D Array of Floats
    TargetMaxs: 1D Array of Floats
    // For classification
    ClassMap: Map of String to Integer (e.g., {"setosa": 0, "versicolor": 1})

Structure ModelData:
    NeuralNetwork: The NeuralNetwork structure from Part 1
    InputMins: 1D Array of Floats
    InputMaxs: 1D Array of Floats
    TargetMins: 1D Array of Floats
    TargetMaxs: 1D Array of Floats
    ClassMap: Map of String to Integer
```

### CSV Loading and Preprocessing

This is a critical step that prepares the data for the network.

**Pseudocode for `LoadCSV`:**

```
FUNCTION LoadCSV(filePath, splitRatio):
    records = read_all_rows_from_csv(filePath)
    header = records[0]
    data_rows = records[1:]

    // --- 1. Detect Task Type (Regression vs. Classification) ---
    // Check if the last column is a number or a string
    isClassification = IS_STRING(data_rows[0][last_column_index])

    // --- 2. Preprocess Data ---
    IF isClassification:
        // Build a map of class names to integer indices
        classMap = {}
        classIndex = 0
        FOR row in data_rows:
            className = row[last_column_index]
            IF className not in classMap:
                classMap[className] = classIndex
                classIndex += 1
        outputSize = length(classMap)
    ELSE:
        outputSize = 1

    inputSize = length(header) - 1

    // --- 3. Normalize Inputs (and Targets for Regression) ---
    // Find min/max for each input column
    inputMins, inputMaxs = find_min_max(data_rows, columns 0 to inputSize-1)
    IF NOT isClassification:
        // Find min/max for target column
        targetMins, targetMaxs = find_min_max(data_rows, column inputSize)

    // --- 4. Build Input and Target Arrays ---
    inputs = []
    targets = []
    FOR row in data_rows:
        // Normalize input features
        inputRow = normalize(row[0 to inputSize-1], inputMins, inputMaxs)
        add inputRow to inputs

        // Create target vector
        IF isClassification:
            // One-hot encode the target
            targetRow = new 1D_Array(outputSize) // All zeros
            className = row[last_column_index]
            targetRow[classMap[className]] = 1.0
        ELSE:
            // Normalize the target value
            targetRow = normalize(row[inputSize], targetMins, targetMaxs)

        add targetRow to targets

    // --- 5. Finalize Dataset ---
    Shuffle(inputs, targets) // Shuffle them in unison
    trainInputs, trainTargets, testInputs, testTargets = SplitData(inputs,
targets, splitRatio)

    RETURN new Dataset(...)
```

### Model Serialization

The `ModelData` structure should be serialized to a file format like JSON. This
allows you to save a trained network and all the necessary preprocessing data
(mins, maxs, classMap) to use it for predictions later.

---

## Part 3: Terminal User Interface (TUI)

This component is highly dependent on the target language and its libraries.

### Recommendations

- **Find a TUI Library:** Look for a library in your target language that can
handle terminal input, styling, and screen rendering. Examples: `ncurses`
(C/C++), `prompt-toolkit` or `curses` (Python), `tui-rs` (Rust).
- **State Machine:** The UI can be modeled as a state machine with the
following states:
  - `MainMenu`: Choose to train or predict.
  - `TrainingForm`: Enter training parameters (CSV file, network layers, epochs,
  etc.).
  - `TrainingInProgress`: Display progress (epoch, loss).
  - `PredictionForm`: Select a saved model and enter input data.
  - `PredictionResult`: Display the model's prediction.
  - `SaveModelForm`: Prompt to save the newly trained model.
- **Asynchronous Operations:** Training the network can be slow. It's best to
run the training loop in a separate thread or asynchronous task to keep the UI
responsive (e.g., to handle a "cancel" button). The training loop can send
progress messages back to the main UI thread for display.

The core responsibility of the TUI is to collect configuration strings from the
user, convert them to the appropriate numbers and arrays, and then call the
`Train` or `Predict` functions from the other components.

