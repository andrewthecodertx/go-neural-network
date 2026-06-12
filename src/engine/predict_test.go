package engine

import (
	"math"
	"testing"

	"go-neuralnetwork/src/data"
	"go-neuralnetwork/src/neuralnetwork"
)

func TestParseInputVector(t *testing.T) {
	got, err := ParseInputVector(" 1.0, 2.5 ,3 ")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []float64{1.0, 2.5, 3.0}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("index %d = %v, want %v", i, got[i], want[i])
		}
	}

	if _, err := ParseInputVector("1,two,3"); err == nil {
		t.Error("expected error for non-numeric input")
	}
}

// buildRegressionModel constructs a trivial model with a known normalization
// range so Predict's normalize -> feedforward -> denormalize path can be
// checked against hand-computed values.
func buildRegressionModel(t *testing.T) *data.ModelData {
	t.Helper()
	// Single linear neuron, identity-ish: 1 input, 1 output.
	nn := neuralnetwork.InitNetwork(1, []int{}, 1, []string{}, "linear")
	return &data.ModelData{
		NN:         nn,
		InputMins:  []float64{0},
		InputMaxs:  []float64{10},
		TargetMins: []float64{0},
		TargetMaxs: []float64{100},
	}
}

func TestPredictInputCountValidation(t *testing.T) {
	md := buildRegressionModel(t)
	if _, err := Predict(md, []float64{1, 2}); err == nil {
		t.Error("expected error when input count != NumInputs")
	}
}

func TestPredictRegressionDenormalizes(t *testing.T) {
	md := buildRegressionModel(t)
	// Force a known output weight so feedforward is deterministic: out = w*x_norm.
	md.NN.OutputWeights[0][0] = 1.0
	md.NN.OutputBiases[0] = 0.0

	// raw input 5 -> normalized 0.5 -> output 0.5 -> denorm 0.5*100 = 50.
	res, err := Predict(md, []float64{5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.IsClassification {
		t.Fatal("expected regression result")
	}
	if math.Abs(res.Value-50.0) > 1e-9 {
		t.Errorf("value = %v, want 50", res.Value)
	}
}

func TestPredictClassification(t *testing.T) {
	nn := neuralnetwork.InitNetwork(1, []int{}, 2, []string{}, "sigmoid")
	// Make class index 1 win regardless of input.
	nn.OutputWeights[0][0] = 0
	nn.OutputWeights[1][0] = 0
	nn.OutputBiases[0] = -10
	nn.OutputBiases[1] = 10

	md := &data.ModelData{
		NN:        nn,
		InputMins: []float64{0},
		InputMaxs: []float64{1},
		ClassMap:  map[string]int{"a": 0, "b": 1},
	}

	res, err := Predict(md, []float64{0.5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.IsClassification {
		t.Fatal("expected classification result")
	}
	if res.Class != "b" {
		t.Errorf("class = %q, want %q", res.Class, "b")
	}
}
