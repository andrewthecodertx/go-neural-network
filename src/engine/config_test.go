package engine

import "testing"

func TestParseTrainConfigDefaults(t *testing.T) {
	cfg, err := ParseTrainConfig(RawTrainConfig{CSVPath: "datasets/iris.csv"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Epochs != DefaultEpochs {
		t.Errorf("epochs = %d, want %d", cfg.Epochs, DefaultEpochs)
	}
	if cfg.LearningRate != DefaultLearningRate {
		t.Errorf("lr = %v, want %v", cfg.LearningRate, DefaultLearningRate)
	}
	if cfg.OutputActivation != DefaultOutputActivation {
		t.Errorf("output activation = %q, want %q", cfg.OutputActivation, DefaultOutputActivation)
	}
	if len(cfg.HiddenLayers) != 2 || cfg.HiddenLayers[0] != 20 || cfg.HiddenLayers[1] != 20 {
		t.Errorf("hidden layers = %v, want [20 20]", cfg.HiddenLayers)
	}
	if cfg.SplitRatio != DefaultSplitRatio {
		t.Errorf("split ratio = %v, want %v", cfg.SplitRatio, DefaultSplitRatio)
	}
}

func TestParseTrainConfigExplicit(t *testing.T) {
	cfg, err := ParseTrainConfig(RawTrainConfig{
		CSVPath:           "datasets/iris.csv",
		HiddenLayers:      "8, 4",
		HiddenActivations: "relu, tanh",
		OutputActivation:  "sigmoid",
		Epochs:            "500",
		LearningRate:      "0.05",
		ErrorGoal:         "0.01",
		SplitRatio:        "0.7",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(cfg.HiddenLayers) != 2 || cfg.HiddenLayers[0] != 8 || cfg.HiddenLayers[1] != 4 {
		t.Errorf("hidden layers = %v, want [8 4]", cfg.HiddenLayers)
	}
	if cfg.HiddenActivations[1] != "tanh" {
		t.Errorf("activations = %v, want second 'tanh'", cfg.HiddenActivations)
	}
	if cfg.Epochs != 500 || cfg.LearningRate != 0.05 || cfg.SplitRatio != 0.7 {
		t.Errorf("got epochs=%d lr=%v split=%v", cfg.Epochs, cfg.LearningRate, cfg.SplitRatio)
	}
}

func TestParseTrainConfigErrors(t *testing.T) {
	cases := []struct {
		name string
		raw  RawTrainConfig
	}{
		{"no dataset", RawTrainConfig{}},
		{"bad layers", RawTrainConfig{CSVPath: "x.csv", HiddenLayers: "10,abc"}},
		{"layer/activation mismatch", RawTrainConfig{CSVPath: "x.csv", HiddenLayers: "10,10,10", HiddenActivations: "relu,relu"}},
		{"bad epochs", RawTrainConfig{CSVPath: "x.csv", Epochs: "lots"}},
		{"zero epochs", RawTrainConfig{CSVPath: "x.csv", Epochs: "0"}},
		{"bad lr", RawTrainConfig{CSVPath: "x.csv", LearningRate: "fast"}},
		{"split out of range", RawTrainConfig{CSVPath: "x.csv", SplitRatio: "1.5"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseTrainConfig(tc.raw); err == nil {
				t.Errorf("expected error for %s, got nil", tc.name)
			}
		})
	}
}
