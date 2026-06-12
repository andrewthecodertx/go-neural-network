package engine

import (
	"fmt"
	"strconv"
	"strings"

	"go-neuralnetwork/src/data"
)

// PredictionResult is the outcome of running a loaded model on a single input.
// Exactly one of Class / Value is meaningful, indicated by IsClassification.
type PredictionResult struct {
	IsClassification bool
	Class            string
	Value            float64
}

// LoadModelForPrediction loads a saved model and rebinds its activation
// functions (which are not serialized).
func LoadModelForPrediction(path string) (*data.ModelData, error) {
	md, err := data.LoadModel(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}
	if err := md.NN.SetActivationFunctions(); err != nil {
		return nil, fmt.Errorf("failed to set activation functions: %w", err)
	}
	return md, nil
}

// Predict normalizes the raw input using the model's stored min/max values,
// runs a feedforward pass, and interprets the output as a class label
// (classification) or a denormalized value (regression).
func Predict(md *data.ModelData, rawInput []float64) (PredictionResult, error) {
	if len(rawInput) != md.NN.NumInputs {
		return PredictionResult{}, fmt.Errorf(
			"expected %d input values, but got %d", md.NN.NumInputs, len(rawInput))
	}

	normalized := make([]float64, md.NN.NumInputs)
	for i, val := range rawInput {
		rangeVal := md.InputMaxs[i] - md.InputMins[i]
		if rangeVal == 0 {
			normalized[i] = 0
		} else {
			normalized[i] = (val - md.InputMins[i]) / rangeVal
		}
	}

	_, output := md.NN.FeedForward(normalized)

	if md.ClassMap != nil {
		maxIndex := argMax(output)
		for class, index := range md.ClassMap {
			if index == maxIndex {
				return PredictionResult{IsClassification: true, Class: class}, nil
			}
		}
		return PredictionResult{}, fmt.Errorf("could not determine class from prediction")
	}

	denorm := output[0]*(md.TargetMaxs[0]-md.TargetMins[0]) + md.TargetMins[0]
	return PredictionResult{Value: denorm}, nil
}

// ParseInputVector parses a comma-separated string of floats into a slice.
func ParseInputVector(s string) ([]float64, error) {
	parts := strings.Split(strings.TrimSpace(s), ",")
	out := make([]float64, len(parts))
	for i, p := range parts {
		val, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid input value %q: %w", strings.TrimSpace(p), err)
		}
		out[i] = val
	}
	return out, nil
}

func argMax(xs []float64) int {
	maxIndex := -1
	maxVal := -1.0
	for i, v := range xs {
		if v > maxVal {
			maxVal = v
			maxIndex = i
		}
	}
	return maxIndex
}
