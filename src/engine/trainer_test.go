package engine

import (
	"os"
	"path/filepath"
	"testing"
)

// resolveProjectRoot walks up from the test's working directory to find the
// repo root (where datasets/ lives), so the end-to-end test works regardless
// of where `go test` is invoked from.
func resolveProjectRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "datasets", "iris.csv")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skip("could not locate datasets/iris.csv; skipping end-to-end test")
		}
		dir = parent
	}
}

// TestTrainAndEvaluateIris drives the entire pipeline through the engine alone:
// load CSV -> build network -> train -> evaluate. This is the test that was
// impossible before the core was decoupled from the TUI.
func TestTrainAndEvaluateIris(t *testing.T) {
	root := resolveProjectRoot(t)

	cfg, err := ParseTrainConfig(RawTrainConfig{
		CSVPath:           filepath.Join(root, "datasets", "iris.csv"),
		HiddenLayers:      "8",
		HiddenActivations: "relu",
		OutputActivation:  "sigmoid",
		Epochs:            "300",
		LearningRate:      "0.1",
		ErrorGoal:         "0",
	})
	if err != nil {
		t.Fatalf("config: %v", err)
	}

	handles, err := StartTraining(cfg)
	if err != nil {
		t.Fatalf("start training: %v", err)
	}

	// Drain progress until the channel closes.
	epochs := 0
	var lastLoss float64
	for loss := range handles.Progress {
		lastLoss = loss
		epochs++
	}
	if epochs == 0 {
		t.Fatal("no epochs reported")
	}

	result := <-handles.Done
	if result.Model == nil || result.TestData == nil {
		t.Fatal("incomplete training result")
	}

	eval := Evaluate(result)
	if !eval.IsClassification {
		t.Fatal("iris should be classified as a classification task")
	}
	// A tiny MLP on iris should comfortably clear chance (3 classes => ~0.33).
	if eval.Accuracy < 0.6 {
		t.Errorf("accuracy = %.3f, expected > 0.6 after %d epochs (last loss %.4f)",
			eval.Accuracy, epochs, lastLoss)
	}
}
