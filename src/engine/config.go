// Package engine owns the orchestration of training, prediction, and model
// persistence. It depends only on the data and neuralnetwork packages, so the
// full pipeline can be driven from any frontend (TUI, CLI, HTTP) and tested
// without a UI.
package engine

import (
	"fmt"
	"strconv"
	"strings"
)

// Defaults applied when a field is left blank.
const (
	DefaultHiddenLayers      = "20,20"
	DefaultHiddenActivations = "relu,relu"
	DefaultOutputActivation  = "linear"
	DefaultEpochs            = 1000
	DefaultLearningRate      = 0.001
	DefaultErrorGoal         = 0.001
	DefaultSplitRatio        = 0.8
)

// TrainConfig is the validated, typed configuration for a training run.
type TrainConfig struct {
	CSVPath           string
	HiddenLayers      []int
	HiddenActivations []string
	OutputActivation  string
	Epochs            int
	LearningRate      float64
	ErrorGoal         float64
	SplitRatio        float64
	EnableViz         bool
}

// RawTrainConfig holds the unparsed string inputs from a frontend form. It maps
// directly onto what a user types so the parsing/validation lives here instead
// of in the UI layer.
type RawTrainConfig struct {
	CSVPath           string
	HiddenLayers      string
	HiddenActivations string
	OutputActivation  string
	Epochs            string
	LearningRate      string
	ErrorGoal         string
	SplitRatio        string
	EnableViz         bool
}

// ParseTrainConfig converts raw string inputs into a validated TrainConfig,
// applying defaults for blank fields.
func ParseTrainConfig(raw RawTrainConfig) (TrainConfig, error) {
	cfg := TrainConfig{
		CSVPath:          strings.TrimSpace(raw.CSVPath),
		OutputActivation: orDefaultStr(raw.OutputActivation, DefaultOutputActivation),
		SplitRatio:       DefaultSplitRatio,
		EnableViz:        raw.EnableViz,
	}

	if cfg.CSVPath == "" {
		return TrainConfig{}, fmt.Errorf("no dataset selected")
	}

	layers, err := parseIntList(orDefaultStr(raw.HiddenLayers, DefaultHiddenLayers))
	if err != nil {
		return TrainConfig{}, fmt.Errorf("invalid hidden layers: %w", err)
	}
	cfg.HiddenLayers = layers

	cfg.HiddenActivations = splitTrim(orDefaultStr(raw.HiddenActivations, DefaultHiddenActivations))
	if len(cfg.HiddenActivations) != len(cfg.HiddenLayers) {
		return TrainConfig{}, fmt.Errorf(
			"hidden activations (%d) must match hidden layers (%d)",
			len(cfg.HiddenActivations), len(cfg.HiddenLayers))
	}

	cfg.Epochs, err = parseIntDefault(raw.Epochs, DefaultEpochs)
	if err != nil {
		return TrainConfig{}, fmt.Errorf("invalid epochs: %w", err)
	}
	if cfg.Epochs <= 0 {
		return TrainConfig{}, fmt.Errorf("epochs must be positive")
	}

	cfg.LearningRate, err = parseFloatDefault(raw.LearningRate, DefaultLearningRate)
	if err != nil {
		return TrainConfig{}, fmt.Errorf("invalid learning rate: %w", err)
	}

	cfg.ErrorGoal, err = parseFloatDefault(raw.ErrorGoal, DefaultErrorGoal)
	if err != nil {
		return TrainConfig{}, fmt.Errorf("invalid error goal: %w", err)
	}

	if r := strings.TrimSpace(raw.SplitRatio); r != "" {
		cfg.SplitRatio, err = strconv.ParseFloat(r, 64)
		if err != nil {
			return TrainConfig{}, fmt.Errorf("invalid split ratio: %w", err)
		}
		if cfg.SplitRatio <= 0 || cfg.SplitRatio >= 1 {
			return TrainConfig{}, fmt.Errorf("split ratio must be between 0 and 1")
		}
	}

	return cfg, nil
}

func orDefaultStr(v, def string) string {
	if strings.TrimSpace(v) == "" {
		return def
	}
	return v
}

func splitTrim(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, len(parts))
	for i, p := range parts {
		out[i] = strings.TrimSpace(p)
	}
	return out
}

func parseIntList(s string) ([]int, error) {
	parts := splitTrim(s)
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		out = append(out, n)
	}
	return out, nil
}

func parseIntDefault(s string, def int) (int, error) {
	if strings.TrimSpace(s) == "" {
		return def, nil
	}
	return strconv.Atoi(strings.TrimSpace(s))
}

func parseFloatDefault(s string, def float64) (float64, error) {
	if strings.TrimSpace(s) == "" {
		return def, nil
	}
	return strconv.ParseFloat(strings.TrimSpace(s), 64)
}
