package engine

import (
	"fmt"

	"go-neuralnetwork/src/data"
	"go-neuralnetwork/src/neuralnetwork"
)

// TrainResult bundles a finished training run: the persistable model and the
// held-out test set used to evaluate it.
type TrainResult struct {
	Model    *data.ModelData
	TestData *data.Dataset
}

// TrainHandles exposes the live channels of an in-flight training run so a
// frontend can stream progress and (optionally) activation snapshots without
// the engine knowing anything about the consumer.
type TrainHandles struct {
	// Progress emits the average error after each epoch. Closed when training ends.
	Progress <-chan float64
	// Viz emits activation snapshots (hidden layers + output) periodically. It
	// is nil when visualization is disabled. Closed when training ends.
	Viz <-chan [][]float64
	// Done emits exactly once with the final result after training completes.
	Done <-chan TrainResult
	// Network is the network being trained, exposed for visualizers that need
	// to read its topology/weights while training runs.
	Network *neuralnetwork.NeuralNetwork
}

// StartTraining loads the dataset, builds the network, and runs training in a
// background goroutine. It returns immediately with channels the caller can use
// to observe progress. The caller is responsible for draining Progress (and Viz
// if non-nil) until they close, then reading Done.
func StartTraining(cfg TrainConfig) (*TrainHandles, error) {
	dataset, err := data.LoadCSV(cfg.CSVPath, cfg.SplitRatio)
	if err != nil {
		return nil, fmt.Errorf("failed to load CSV data: %w", err)
	}

	nn := neuralnetwork.InitNetwork(
		dataset.InputSize,
		cfg.HiddenLayers,
		dataset.OutputSize,
		cfg.HiddenActivations,
		cfg.OutputActivation,
	)

	progressChan := make(chan float64)
	doneChan := make(chan TrainResult, 1)

	var vizChan chan [][]float64
	if cfg.EnableViz {
		vizChan = make(chan [][]float64)
	}

	go func() {
		nn.Train(
			dataset.TrainInputs, dataset.TrainTargets,
			cfg.Epochs, cfg.LearningRate, cfg.ErrorGoal,
			progressChan, vizChan,
		)
		doneChan <- TrainResult{
			Model: &data.ModelData{
				NN:         nn,
				InputMins:  dataset.InputMins,
				InputMaxs:  dataset.InputMaxs,
				TargetMins: dataset.TargetMins,
				TargetMaxs: dataset.TargetMaxs,
				ClassMap:   dataset.ClassMap,
			},
			TestData: dataset,
		}
	}()

	h := &TrainHandles{
		Progress: progressChan,
		Done:     doneChan,
		Network:  nn,
	}
	if vizChan != nil {
		h.Viz = vizChan
	}
	return h, nil
}

// Evaluate runs the model against its held-out test set.
func Evaluate(result TrainResult) data.EvaluationResult {
	return data.Evaluate(result.Model, result.TestData.TestInputs, result.TestData.TestTargets)
}
