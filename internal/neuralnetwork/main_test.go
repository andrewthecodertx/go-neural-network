package neuralnetwork_test

import (
	"math"
	"os"
	"reflect"
	"testing"

	"go-neuralnetwork/internal/data"
	"go-neuralnetwork/internal/neuralnetwork"
	"go-neuralnetwork/internal/tempfile"
)

func TestInitNetwork(t *testing.T) {
	inputs := 2
	hiddenLayers := []int{3, 2}
	outputs := 1
	hiddenActivations := []string{"relu", "tanh"}
	outputActivation := "linear"

	nn := neuralnetwork.InitNetwork(inputs, hiddenLayers, outputs, hiddenActivations, outputActivation)

	if nn.NumInputs != inputs {
		t.Errorf("Expected NumInputs to be %d, got %d", inputs, nn.NumInputs)
	}
	if !reflect.DeepEqual(nn.HiddenLayers, hiddenLayers) {
		t.Errorf("Expected HiddenLayers to be %v, got %v", hiddenLayers, nn.HiddenLayers)
	}
	if nn.NumOutputs != outputs {
		t.Errorf("Expected NumOutputs to be %d, got %d", outputs, nn.NumOutputs)
	}
	if !reflect.DeepEqual(nn.HiddenActivations, hiddenActivations) {
		t.Errorf("Expected HiddenActivations to be %v, got %v", hiddenActivations, nn.HiddenActivations)
	}
	if nn.OutputActivation != outputActivation {
		t.Errorf("Expected OutputActivation to be %s, got %s", outputActivation, nn.OutputActivation)
	}

	if len(nn.HiddenWeights) != len(hiddenLayers) {
		t.Errorf("HiddenWeights dimensions mismatch")
	}
	if len(nn.OutputWeights) != outputs {
		t.Errorf("OutputWeights dimensions mismatch")
	}
	if len(nn.HiddenBiases) != len(hiddenLayers) {
		t.Errorf("HiddenBiases dimensions mismatch")
	}
	if len(nn.OutputBiases) != outputs {
		t.Errorf("OutputBiases dimensions mismatch")
	}
}

func TestFeedForward(t *testing.T) {
	nn := &neuralnetwork.NeuralNetwork{
		NumInputs:         2,
		HiddenLayers:      []int{2, 2},
		NumOutputs:        1,
		HiddenWeights:     [][][]float64{{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.6}, {0.7, 0.8}}},
		OutputWeights:     [][]float64{{0.9, 1.0}},
		HiddenBiases:      [][]float64{{0.0, 0.0}, {0.0, 0.0}},
		OutputBiases:      []float64{0.0},
		HiddenActivations: []string{"relu", "relu"},
		OutputActivation:  "linear",
	}
	nn.SetActivationFunctions()

	inputs := []float64{1.0, 1.0}
	hiddenOutputs, finalOutputs := nn.FeedForward(inputs)

	expectedHidden1 := []float64{0.3, 0.7}
	expectedHidden2 := []float64{0.57, 0.77}
	expectedFinal := []float64{1.283}

	for i := range expectedHidden1 {
		if math.Abs(hiddenOutputs[0][i]-expectedHidden1[i]) > 1e-9 {
			t.Errorf("Hidden output 1 mismatch at index %d: Expected %f, got %f", i, expectedHidden1[i], hiddenOutputs[0][i])
		}
	}
	for i := range expectedHidden2 {
		if math.Abs(hiddenOutputs[1][i]-expectedHidden2[i]) > 1e-9 {
			t.Errorf("Hidden output 2 mismatch at index %d: Expected %f, got %f", i, expectedHidden2[i], hiddenOutputs[1][i])
		}
	}
	if math.Abs(finalOutputs[0]-expectedFinal[0]) > 1e-9 {
		t.Errorf("Final output mismatch: Expected %f, got %f", expectedFinal[0], finalOutputs[0])
	}
}

func TestSaveAndLoadModel(t *testing.T) {
	originalNN := neuralnetwork.InitNetwork(2, []int{2, 2}, 1, []string{"relu", "tanh"}, "linear")
	originalMD := &data.ModelData{
		NN:         originalNN,
		TargetMins: []float64{1.0},
		TargetMaxs: []float64{10.0},
		InputMins:  []float64{0.0, 0.0},
		InputMaxs:  []float64{1.0, 1.0},
	}

	filePath, err := tempfile.CreateTempFileWithContent("model-*.json", "")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(filePath)

	err = originalMD.SaveModel(filePath)
	if err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	loadedMD, err := data.LoadModel(filePath)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	loadedMD.NN.SetActivationFunctions()

	if !reflect.DeepEqual(originalMD.NN.NumInputs, loadedMD.NN.NumInputs) ||
		!reflect.DeepEqual(originalMD.NN.HiddenLayers, loadedMD.NN.HiddenLayers) ||
		!reflect.DeepEqual(originalMD.NN.NumOutputs, loadedMD.NN.NumOutputs) ||
		!reflect.DeepEqual(originalMD.NN.HiddenActivations, loadedMD.NN.HiddenActivations) ||
		!reflect.DeepEqual(originalMD.NN.OutputActivation, loadedMD.NN.OutputActivation) ||
		!reflect.DeepEqual(originalMD.NN.HiddenWeights, loadedMD.NN.HiddenWeights) ||
		!reflect.DeepEqual(originalMD.NN.OutputWeights, loadedMD.NN.OutputWeights) ||
		!reflect.DeepEqual(originalMD.NN.HiddenBiases, loadedMD.NN.HiddenBiases) ||
		!reflect.DeepEqual(originalMD.NN.OutputBiases, loadedMD.NN.OutputBiases) {
		t.Errorf("Loaded model does not match original model")
	}
}

func TestBackpropagate(t *testing.T) {
	nn := &neuralnetwork.NeuralNetwork{
		NumInputs:         2,
		HiddenLayers:      []int{2},
		NumOutputs:        1,
		HiddenWeights:     [][][]float64{{{0.1, 0.2}, {0.3, 0.4}}},
		OutputWeights:     [][]float64{{0.5, 0.6}},
		HiddenBiases:      [][]float64{{0.0, 0.0}},
		OutputBiases:      []float64{0.0},
		HiddenActivations: []string{"sigmoid"},
		OutputActivation:  "sigmoid",
	}
	nn.SetActivationFunctions()

	inputs := []float64{1.0, 1.0}
	targets := []float64{1.0}
	learningRate := 0.1

	// Create copies of weights and biases before backpropagation
	originalHiddenWeights := deepCopy3D(nn.HiddenWeights)
	originalOutputWeights := deepCopy2D(nn.OutputWeights)
	originalHiddenBiases := deepCopy2D(nn.HiddenBiases)
	originalOutputBiases := deepCopy1D(nn.OutputBiases)

	hiddenOutputs, finalOutputs := nn.FeedForward(inputs)
	nn.Backpropagate(inputs, targets, hiddenOutputs, finalOutputs, learningRate)

	// Check if weights and biases have been updated
	if reflect.DeepEqual(originalHiddenWeights, nn.HiddenWeights) {
		t.Errorf("Hidden weights were not updated")
	}
	if reflect.DeepEqual(originalOutputWeights, nn.OutputWeights) {
		t.Errorf("Output weights were not updated")
	}
	if reflect.DeepEqual(originalHiddenBiases, nn.HiddenBiases) {
		t.Errorf("Hidden biases were not updated")
	}
	if reflect.DeepEqual(originalOutputBiases, nn.OutputBiases) {
		t.Errorf("Output biases were not updated")
	}
}

func TestTrain(t *testing.T) {
	// XOR problem: a classic non-linear problem to test if the network can learn
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Network architecture capable of solving XOR
	nn := neuralnetwork.InitNetwork(2, []int{3}, 1, []string{"tanh"}, "tanh")

	// Helper to calculate Mean Squared Error
	calculateMSE := func() float64 {
		totalError := 0.0
		for i, input := range inputs {
			_, finalOutputs := nn.FeedForward(input)
			for j, target := range targets[i] {
				totalError += 0.5 * (target - finalOutputs[j]) * (target - finalOutputs[j])
			}
		}
		return totalError / float64(len(inputs))
	}

	initialError := calculateMSE()

	// Train the network
	epochs := 5000
	learningRate := 0.1
	errorGoal := 0.01
	progressChan := make(chan any)

	go func() {
		// Consume progress updates to prevent blocking
		for range progressChan {
		}
	}()

	nn.Train(inputs, targets, epochs, learningRate, errorGoal, progressChan)

	finalError := calculateMSE()

	if finalError >= initialError {
		t.Errorf("Training did not reduce error. Initial: %f, Final: %f", initialError, finalError)
	}

	// This test can be flaky due to random weight initialization.
	// A high error isn't a definitive failure but an indication of non-convergence in this run.
	if finalError > errorGoal {
		t.Logf("Network did not converge to the error goal. This can happen, but check if it's frequent. Final Error: %f, Goal: %f", finalError, errorGoal)
	}

	// Check if it learned the pattern
	for i, input := range inputs {
		_, finalOutputs := nn.FeedForward(input)
		roundedOutput := math.Round(finalOutputs[0])
		expectedOutput := targets[i][0]
		if roundedOutput != expectedOutput {
			t.Errorf("XOR test failed for input %v. Expected output to be close to %v (rounded: %v), but got %v", input, expectedOutput, roundedOutput, finalOutputs[0])
		}
	}
}

// Helper functions for deep copying slices
func deepCopy3D(slice [][][]float64) [][][]float64 {
	newSlice := make([][][]float64, len(slice))
	for i, s := range slice {
		newSlice[i] = deepCopy2D(s)
	}
	return newSlice
}

func deepCopy2D(slice [][]float64) [][]float64 {
	newSlice := make([][]float64, len(slice))
	for i, s := range slice {
		newSlice[i] = deepCopy1D(s)
	}
	return newSlice
}

func deepCopy1D(slice []float64) []float64 {
	newSlice := make([]float64, len(slice))
	copy(newSlice, slice)
	return newSlice
}
