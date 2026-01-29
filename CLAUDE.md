# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

- **Run**: `go run .`
- **Build**: `go build`
- **Test all**: `go test ./...`
- **Test single function**: `go test -run TestFunctionName ./path/to/package`
- **Format**: `gofmt -w .`
- **Check formatting**: `gofmt -l .`
- **Static analysis**: `go vet ./...`

## Architecture

This is a feed-forward neural network implementation in Go with a terminal UI. The entry point (`main.go`) simply calls `tui.Start()`.

### Package Overview

- **`internal/tui`** — Bubbletea-based terminal UI using an Elm-architecture state machine. Manages the full user workflow: dataset selection, training configuration, progress display, evaluation, prediction, and model save/load. Training runs asynchronously via goroutines with channel-based progress reporting back to the UI.

- **`internal/neuralnetwork`** — Core network: weight initialization (He init), feed-forward propagation, backpropagation with gradient descent, and MSE loss. Activation functions (ReLU, Sigmoid, Tanh, Linear) implement the `Activation` interface with `Activate` and `Derivative` methods.

- **`internal/data`** — CSV loading with automatic detection of regression vs classification tasks. Handles min-max normalization, one-hot encoding for classification targets, train/test splitting, and JSON serialization of trained models (network weights + normalization metadata).

- **`internal/visualization`** — SDL2-based real-time visualization of network structure during training. Shows color-coded node activations and weight connections. Requires SDL2 system libraries.

### Key Data Flow

**Training**: TUI form → `data.LoadCSV()` → `neuralnetwork.InitNetwork()` → `nn.Train()` (with epoch progress via channels) → evaluate on test set → save as JSON to `saved_models/`

**Prediction**: Load model JSON → restore network + normalization params → normalize input → `nn.FeedForward()` → denormalize output or map to class label

### Datasets

CSV datasets live in `assets/`. The data loader auto-detects whether the last column is categorical (classification) or numeric (regression).

## Code Style

- Follow standard Go conventions; use `gofmt` for formatting
- Group imports: stdlib, then third-party, then local packages
- Use table-driven tests with `floatTolerance` constants for float comparisons
- Use pointer receivers for methods that modify state
- Use struct tags for JSON serialization (e.g., `json:"numInputs"`)
