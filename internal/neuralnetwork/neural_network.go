package neuralnetwork

import (
	"math"
	"math/rand"
)

// NeuralNetwork represents a multi-layer perceptron.
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
	hiddenActivationFuncs []Activation  `json:"-"`
	outputActivationFunc  Activation    `json:"-"`
}

// InitNetwork initializes a new neural network with random weights and biases.
func InitNetwork(inputs int, hiddenLayers []int, outputs int, hiddenActivations []string, outputActivation string) *NeuralNetwork {
	hiddenWeights := make([][][]float64, len(hiddenLayers))
	hiddenBiases := make([][]float64, len(hiddenLayers))
	prevLayerSize := inputs

	// Initialize hidden layers
	for i, layerSize := range hiddenLayers {
		hiddenWeights[i] = make([][]float64, layerSize)
		initScale := weightInitScale(hiddenActivations[i], prevLayerSize)
		for j := range hiddenWeights[i] {
			hiddenWeights[i][j] = make([]float64, prevLayerSize)
			for k := range hiddenWeights[i][j] {
				hiddenWeights[i][j][k] = rand.NormFloat64() * initScale
			}
		}
		hiddenBiases[i] = make([]float64, layerSize)
		prevLayerSize = layerSize
	}

	// Initialize output layer
	outputWeights := make([][]float64, outputs)
	initScale := weightInitScale(outputActivation, prevLayerSize)
	for i := range outputWeights {
		outputWeights[i] = make([]float64, prevLayerSize)
		for j := range outputWeights[i] {
			outputWeights[i][j] = rand.NormFloat64() * initScale
		}
	}
	outputBiases := make([]float64, outputs)

	nn := &NeuralNetwork{
		NumInputs:         inputs,
		HiddenLayers:      hiddenLayers,
		NumOutputs:        outputs,
		HiddenWeights:     hiddenWeights,
		OutputWeights:     outputWeights,
		HiddenBiases:      hiddenBiases,
		OutputBiases:      outputBiases,
		HiddenActivations: hiddenActivations,
		OutputActivation:  outputActivation,
	}
	nn.SetActivationFunctions()
	return nn
}

// weightInitScale returns the standard deviation for weight initialization based on the activation function.
// He initialization (sqrt(2/fan_in)) is used for ReLU, Xavier/Glorot (sqrt(1/fan_in)) for sigmoid, tanh, and linear.
func weightInitScale(activation string, fanIn int) float64 {
	switch activation {
	case "relu":
		return math.Sqrt(2.0 / float64(fanIn))
	default:
		return math.Sqrt(1.0 / float64(fanIn))
	}
}

// SetActivationFunctions sets the activation functions for each layer.
func (nn *NeuralNetwork) SetActivationFunctions() error {
	nn.hiddenActivationFuncs = make([]Activation, len(nn.HiddenActivations))
	for i, activationName := range nn.HiddenActivations {
		activation, err := GetActivation(activationName)
		if err != nil {
			return err
		}
		nn.hiddenActivationFuncs[i] = activation
	}

	activation, err := GetActivation(nn.OutputActivation)
	if err != nil {
		return err
	}
	nn.outputActivationFunc = activation
	return nil
}

// FeedForward performs the feedforward pass of the neural network.
func (nn *NeuralNetwork) FeedForward(inputs []float64) ([][]float64, []float64) {
	hiddenOutputs := make([][]float64, len(nn.HiddenLayers))
	layerInput := inputs

	// Calculate hidden layer outputs
	for i, layerSize := range nn.HiddenLayers {
		hiddenOutputs[i] = make([]float64, layerSize)
		for j := range hiddenOutputs[i] {
			sum := 0.0
			for k, val := range layerInput {
				sum += val * nn.HiddenWeights[i][j][k]
			}
			hiddenOutputs[i][j] = nn.hiddenActivationFuncs[i].Activate(sum + nn.HiddenBiases[i][j])
		}
		layerInput = hiddenOutputs[i]
	}

	// Calculate final output
	finalOutputs := make([]float64, nn.NumOutputs)
	for i := range finalOutputs {
		sum := 0.0
		for j, val := range layerInput {
			sum += val * nn.OutputWeights[i][j]
		}
		finalOutputs[i] = nn.outputActivationFunc.Activate(sum + nn.OutputBiases[i])
	}

	return hiddenOutputs, finalOutputs
}

// Backpropagate performs the backpropagation algorithm to update the weights and biases of the network.
func (nn *NeuralNetwork) Backpropagate(inputs []float64, targets []float64, hiddenOutputs [][]float64, finalOutputs []float64, learningRate float64) {
	// Calculate output layer errors and deltas
	outputErrors := make([]float64, nn.NumOutputs)
	outputDeltas := make([]float64, nn.NumOutputs)
	for i := range outputErrors {
		outputErrors[i] = targets[i] - finalOutputs[i]
		outputDeltas[i] = outputErrors[i] * nn.outputActivationFunc.Derivative(finalOutputs[i])
	}

	// Calculate hidden layer errors and deltas
	hiddenErrors := make([][]float64, len(nn.HiddenLayers))
	hiddenDeltas := make([][]float64, len(nn.HiddenLayers))
	nextLayerDeltas := outputDeltas
	nextLayerWeights := nn.OutputWeights

	for i := len(nn.HiddenLayers) - 1; i >= 0; i-- {
		layerSize := nn.HiddenLayers[i]
		hiddenErrors[i] = make([]float64, layerSize)
		hiddenDeltas[i] = make([]float64, layerSize)
		for j := range hiddenErrors[i] {
			sum := 0.0
			for k, delta := range nextLayerDeltas {
				sum += delta * nextLayerWeights[k][j]
			}
			hiddenErrors[i][j] = sum
			hiddenDeltas[i][j] = hiddenErrors[i][j] * nn.hiddenActivationFuncs[i].Derivative(hiddenOutputs[i][j])
		}

		if i > 0 {
			nextLayerDeltas = hiddenDeltas[i]
			nextLayerWeights = nn.HiddenWeights[i]
		}
	}

	// Update output weights and biases
	lastHiddenLayerOutput := hiddenOutputs[len(hiddenOutputs)-1]
	for i := range nn.OutputWeights {
		for j, val := range lastHiddenLayerOutput {
			nn.OutputWeights[i][j] += learningRate * outputDeltas[i] * val
		}
		nn.OutputBiases[i] += learningRate * outputDeltas[i]
	}

	// Update hidden weights and biases
	for i := len(nn.HiddenLayers) - 1; i >= 0; i-- {
		var prevLayerOutput []float64
		if i == 0 {
			prevLayerOutput = inputs
		} else {
			prevLayerOutput = hiddenOutputs[i-1]
		}
		for j := range nn.HiddenWeights[i] {
			for k, val := range prevLayerOutput {
				nn.HiddenWeights[i][j][k] += learningRate * hiddenDeltas[i][j] * val
			}
			nn.HiddenBiases[i][j] += learningRate * hiddenDeltas[i][j]
		}
	}
}

// Train trains the neural network using the provided training data, number of epochs, learning rate, and error goal.
// An optional vizChan can be provided for real-time visualization of activations; pass nil to disable.
func (nn *NeuralNetwork) Train(inputs, targets [][]float64, epochs int, learningRate float64, errorGoal float64, progressChan chan<- float64, vizChan chan<- [][]float64) {
	defer close(progressChan)
	if vizChan != nil {
		defer close(vizChan)
	}

	for epoch := range make([]struct{}, epochs) {
		// Shuffle training data each epoch to avoid learning order-dependent patterns
		rand.Shuffle(len(inputs), func(i, j int) {
			inputs[i], inputs[j] = inputs[j], inputs[i]
			targets[i], targets[j] = targets[j], targets[i]
		})

		totalError := 0.0
		var lastActivations [][]float64

		for i := range inputs {
			hiddenOutputs, finalOutputs := nn.FeedForward(inputs[i])
			nn.Backpropagate(inputs[i], targets[i], hiddenOutputs, finalOutputs, learningRate)

			// Store activations for visualization (combine hidden + output)
			if vizChan != nil {
				lastActivations = make([][]float64, len(hiddenOutputs)+1)
				copy(lastActivations, hiddenOutputs)
				lastActivations[len(hiddenOutputs)] = make([]float64, len(finalOutputs))
				copy(lastActivations[len(hiddenOutputs)], finalOutputs)
			}

			// Calculate mean squared error
			for j := range targets[i] {
				totalError += 0.5 * (targets[i][j] - finalOutputs[j]) * (targets[i][j] - finalOutputs[j])
			}
		}
		avgError := totalError / float64(len(inputs))

		// Send progress update
		progressChan <- avgError

		// Send visualization data every few epochs to avoid overwhelming the UI
		if vizChan != nil && epoch%5 == 0 {
			vizChan <- lastActivations
		}

		// Stop training if the error goal is reached
		if avgError < errorGoal {
			break
		}
	}
}
