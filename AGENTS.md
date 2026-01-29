# Agent Instructions for Go Neural Network

## Build/Test Commands
- **Build/Run**: `go run .`
- **Test all packages**: `go test ./...`
- **Test single function**: `go test -run TestFunctionName ./path/to/package`
- **Format code**: `gofmt -w .`
- **Check formatting**: `gofmt -l .`
- **Static analysis**: `go vet ./...`

## Code Style Guidelines

### Naming Conventions
- **Packages**: lowercase, single word (e.g., `neuralnetwork`, `data`, `tui`)
- **Types/Structs**: PascalCase (e.g., `NeuralNetwork`, `Dataset`, `ReLU`)
- **Functions**: PascalCase for exported, camelCase for unexported
- **Variables**: camelCase (e.g., `hiddenLayers`, `numInputs`)
- **Constants**: camelCase or PascalCase based on export
- **Interfaces**: PascalCase (e.g., `Activation`)

### Formatting & Structure
- Use `gofmt` for consistent formatting
- Files must end with newline
- Group imports: standard library, blank line, third-party, blank line, local packages
- Use struct tags for JSON serialization (e.g., `json:"numInputs"`)

### Error Handling
- Return errors from functions that can fail
- Use `defer` for resource cleanup (e.g., file.Close())
- Check errors immediately after operations

### Types & Interfaces
- Define interfaces for activation functions and other abstractions
- Use pointer receivers for methods that modify state
- Export types and functions that need to be used across packages

### Testing
- Use table-driven tests with descriptive names
- Test both success and error cases
- Use `floatTolerance` constants for floating-point comparisons

## Visualization
- SDL2-based real-time neural network visualization available
- Enable with "y" in training form visualization field
- Shows network structure with colored nodes (activation intensity) and connections (weight strength)
- Press 'q' or ESC to exit visualization</content>
<parameter name="filePath">/mnt/internalssd/Projects/go-neuralnetwork/AGENTS.md