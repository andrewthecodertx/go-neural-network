package data

import (
	"encoding/json"
	"math"
	"os"

	"go-neuralnetwork/internal/neuralnetwork"
)

// EvaluationResult holds the results of evaluating a model on test data.
type EvaluationResult struct {
	// Classification metrics
	Accuracy float64
	// Regression metrics
	RMSE     float64
	RSquared float64
	// IsClassification indicates which metrics are relevant
	IsClassification bool
}

// Evaluate runs the model against test data and returns evaluation metrics.
func Evaluate(md *ModelData, testInputs, testTargets [][]float64) EvaluationResult {
	if md.ClassMap != nil {
		return evaluateClassification(md, testInputs, testTargets)
	}
	return evaluateRegression(md, testInputs, testTargets)
}

func evaluateClassification(md *ModelData, testInputs, testTargets [][]float64) EvaluationResult {
	correct := 0
	for i, input := range testInputs {
		_, prediction := md.NN.FeedForward(input)
		maxVal := -1.0
		maxIndex := -1
		for j, val := range prediction {
			if val > maxVal {
				maxVal = val
				maxIndex = j
			}
		}

		actualIndex := -1
		for j, val := range testTargets[i] {
			if val == 1.0 {
				actualIndex = j
				break
			}
		}

		if maxIndex == actualIndex {
			correct++
		}
	}

	accuracy := 0.0
	if len(testInputs) > 0 {
		accuracy = float64(correct) / float64(len(testInputs))
	}
	return EvaluationResult{Accuracy: accuracy, IsClassification: true}
}

func evaluateRegression(md *ModelData, testInputs, testTargets [][]float64) EvaluationResult {
	if len(testInputs) == 0 {
		return EvaluationResult{}
	}

	// Calculate RMSE and R²
	sumSquaredError := 0.0
	sumActual := 0.0
	for i, input := range testInputs {
		_, prediction := md.NN.FeedForward(input)
		for j := range testTargets[i] {
			diff := testTargets[i][j] - prediction[j]
			sumSquaredError += diff * diff
			sumActual += testTargets[i][j]
		}
	}

	n := float64(len(testInputs))
	rmse := math.Sqrt(sumSquaredError / n)

	// R² = 1 - SS_res / SS_tot
	meanActual := sumActual / n
	ssTot := 0.0
	for _, target := range testTargets {
		for _, val := range target {
			diff := val - meanActual
			ssTot += diff * diff
		}
	}

	rSquared := 0.0
	if ssTot > 0 {
		rSquared = 1.0 - (sumSquaredError / ssTot)
	}

	return EvaluationResult{RMSE: rmse, RSquared: rSquared}
}

type ModelData struct {
	NN         *neuralnetwork.NeuralNetwork `json:"neuralNetwork"`
	InputMins  []float64                    `json:"inputMins"`
	InputMaxs  []float64                    `json:"inputMaxs"`
	TargetMins []float64                    `json:"targetMins,omitempty"`
	TargetMaxs []float64                    `json:"targetMaxs,omitempty"`
	ClassMap   map[string]int               `json:"classMap,omitempty"`
}

func (md *ModelData) SaveModel(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(md)
}

func LoadModel(filePath string) (*ModelData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var md ModelData
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&md)
	if err != nil {
		return nil, err
	}
	return &md, nil
}
