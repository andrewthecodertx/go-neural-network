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
		// He initialization for weights
		heInit := math.Sqrt(2.0 / float64(prevLayerSize))
		for j := range hiddenWeights[i] {
			hiddenWeights[i][j] = make([]float64, prevLayerSize)
			for k := range hiddenWeights[i][j] {
				hiddenWeights[i][j][k] = rand.NormFloat64() * heInit
			}
		}
		hiddenBiases[i] = make([]float64, layerSize)
		prevLayerSize = layerSize
	}

	// Initialize output layer
	outputWeights := make([][]float64, outputs)
	// He initialization for weights
	heInitOutput := math.Sqrt(2.0 / float64(prevLayerSize))
	for i := range outputWeights {
		outputWeights[i] = make([]float64, prevLayerSize)
		for j := range outputWeights[i] {
			outputWeights[i][j] = rand.NormFloat64() * heInitOutput
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
func (nn *NeuralNetwork) Train(inputs, targets [][]float64, epochs int, learningRate float64, errorGoal float64, progressChan chan<- any) {
	defer close(progressChan) // Ensure the channel is closed when training is done

	for range make([]struct{}, epochs) {
		totalError := 0.0
		for i := range inputs {
			hiddenOutputs, finalOutputs := nn.FeedForward(inputs[i])
			nn.Backpropagate(inputs[i], targets[i], hiddenOutputs, finalOutputs, learningRate)
			// Calculate mean squared error
			for j := range targets[i] {
				totalError += 0.5 * (targets[i][j] - finalOutputs[j]) * (targets[i][j] - finalOutputs[j])
			}
		}
		avgError := totalError / float64(len(inputs))

		// Send progress update
		progressChan <- avgError

		// Stop training if the error goal is reached
		if avgError < errorGoal {
			break
		}
	}
}
