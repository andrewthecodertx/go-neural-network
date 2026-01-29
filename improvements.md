# Improvements for Go Neural Network

## Neural Network Core

### Training Data Shuffling
No training data shuffling between epochs. The training loop in `neural_network.go:186-206` iterates over the data in the same order every epoch. Stochastic gradient descent benefits significantly from reshuffling between epochs to avoid the network learning order-dependent patterns. The `data.Shuffle` function exists but is only called once before the train/test split.

### Mini-Batch Support
No mini-batch support. Training processes every sample individually (online/stochastic SGD). Adding mini-batch gradient descent would improve training stability and allow averaging gradients across a batch before updating weights, which is standard practice.

### Momentum and Adaptive Learning Rate
No momentum or adaptive learning rate. The weight update rule at `neural_network.go:175` is plain gradient descent (`weight += lr * delta * input`). Adding momentum (SGD+momentum) or implementing Adam would make convergence faster and more reliable, especially for the XOR test which is noted as potentially flaky due to this (`main_test.go:226`).

### Division by Zero Risk
Division by zero risk in normalization. In `csvloader.go:812`, the prediction input normalization does `(val - min) / (max - min)` with no zero-division guard. The training-time loader handles this (`csvloader.go:113`), but the prediction path in `tui.go` doesn't.

### Initialization Strategy
He initialization is used unconditionally. He init (`neural_network.go:33`) is designed for ReLU. When using sigmoid or tanh activations, Xavier/Glorot initialization would be more appropriate. The initialization strategy should match the chosen activation function.

## Architecture / Code Organization

### TUI File Overload
The TUI file is doing too much. `tui.go` is 898 lines handling UI state, training orchestration, evaluation logic, and prediction logic. The training command (`runTraining`, lines 37-163) and prediction command (`runPrediction`, lines 785-838) contain business logic (data loading, evaluation, denormalization) that should live outside the TUI layer. This would also make it possible to add a CLI/non-interactive mode without duplicating logic.

### Evaluation Logic Placement
Evaluation logic is embedded in a message handler. The accuracy calculation at `tui.go:375-406` runs inside an anonymous function returned from the `trainingFinishedMsg` handler. This is hard to test and tightly coupled to the TUI. It should be a method on `NeuralNetwork` or a function in the `data` package.

### Regression Evaluation Metric
No regression evaluation metric. The evaluation code at `tui.go:399-401` explicitly skips regression problems with a comment "we could calculate loss here instead." For regression models, the user sees 0% accuracy, which is misleading. R-squared or RMSE would be useful.

### Duplicate Training Methods
`Train` and `TrainWithVisualization` are near-duplicates. `neural_network.go:183-251` has two training methods with almost identical logic. The visualization channel could be an optional parameter in a single method (it already checks `vizChan != nil`).

## Data Handling

### Redundant CSV Reading
CSV file is read twice for regression. `LoadCSV` in `csvloader.go:43` reads all records, checks the last column type, then if it's numeric, seeks back to the beginning and reads them again (`csvloader.go:67-69`). The records are already in memory — they could be reused.

### Duplicated Normalization Logic
`LoadCSV` and `LoadCSVForClassification` have duplicated normalization logic. Both functions independently implement min-max normalization with nearly identical code. This could be a shared helper.

### Single Output Column Limitation
Hardcoded single output column. `LoadCSV` assumes `outputSize := 1` at `csvloader.go:76`. Multi-output regression isn't supported.

## Concurrency

### Race Condition on Program Field
Race condition on `m.program`. In `tui.go:123` and `tui.go:131`, goroutines call `m.program.Send()` but `m.program` is set after `New()` returns, at `tui.go:891`. If training starts before the program is fully set up, this will nil-pointer panic. The program field should be set before any commands can fire.

### Missing Training Cancellation
No way to cancel training. Pressing 'q' during training (`tui.go:430-433`) changes the UI state to `mainMenu`, but the training goroutine continues running in the background. There's no context or cancellation channel passed to `Train()`. The goroutine keeps sending on `progressChan` to a model that's no longer listening, and the network keeps consuming CPU.

### Unsafe Type Assertion
Unsafe type assertion. At `tui.go:131`, `loss.(float64)` will panic if the channel ever receives a non-float64 value. The `progressChan` is typed as `chan any` when it could simply be `chan float64`.

## Testing

### Missing Classification Tests
No tests for the data package's classification path. `csvloader_test.go` exists but the classification loader (`LoadCSVForClassification`) has no dedicated test coverage.

### Missing Edge Case Tests
No tests for edge cases. Networks with a single hidden layer, zero hidden layers, or single input/output aren't explicitly tested. The backpropagation test only verifies that weights changed, not that they changed in the correct direction.

## Visualization

### Broken SDL2 Integration
SDL2 integration is currently broken. The visualization code in `tui.go:137-160` is entirely commented out with a TODO. The visualization package exists and compiles but isn't wired in. This also means the SDL2 dependency in `go.mod` is dead weight for anyone building without SDL2 libraries installed — it could be behind a build tag.
