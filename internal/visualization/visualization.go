package visualization

import (
	"fmt"
	"math"

	"go-neuralnetwork/internal/neuralnetwork"

	"github.com/veandco/go-sdl2/sdl"
)

// NetworkVisualizer handles the SDL2 visualization of the neural network
type NetworkVisualizer struct {
	window   *sdl.Window
	renderer *sdl.Renderer
	running  bool
	width    int32
	height   int32
}

// NewNetworkVisualizer creates a new visualizer with SDL2 window
func NewNetworkVisualizer(width, height int32) (*NetworkVisualizer, error) {
	if err := sdl.Init(sdl.INIT_VIDEO); err != nil {
		return nil, fmt.Errorf("failed to initialize SDL: %v", err)
	}

	window, err := sdl.CreateWindow("Neural Network Visualization",
		sdl.WINDOWPOS_UNDEFINED, sdl.WINDOWPOS_UNDEFINED,
		width, height, sdl.WINDOW_SHOWN|sdl.WINDOW_RESIZABLE)
	if err != nil {
		sdl.Quit()
		return nil, fmt.Errorf("failed to create window: %v", err)
	}

	renderer, err := sdl.CreateRenderer(window, -1, sdl.RENDERER_ACCELERATED)
	if err != nil {
		window.Destroy()
		sdl.Quit()
		return nil, fmt.Errorf("failed to create renderer: %v", err)
	}

	return &NetworkVisualizer{
		window:   window,
		renderer: renderer,
		running:  true,
		width:    width,
		height:   height,
	}, nil
}

// Close cleans up SDL2 resources
func (nv *NetworkVisualizer) Close() {
	if nv.renderer != nil {
		nv.renderer.Destroy()
	}
	if nv.window != nil {
		nv.window.Destroy()
	}
	sdl.Quit()
}

// IsRunning returns whether the visualization is still active
func (nv *NetworkVisualizer) IsRunning() bool {
	return nv.running
}

// HandleEvents processes SDL events
func (nv *NetworkVisualizer) HandleEvents() {
	for event := sdl.PollEvent(); event != nil; event = sdl.PollEvent() {
		switch e := event.(type) {
		case *sdl.QuitEvent:
			nv.running = false
		case *sdl.KeyboardEvent:
			if e.Type == sdl.KEYDOWN && e.Keysym.Sym == sdl.K_ESCAPE {
				nv.running = false
			}
		}
	}
}

// Clear clears the screen with a background color
func (nv *NetworkVisualizer) Clear() {
	nv.renderer.SetDrawColor(20, 20, 30, 255) // Dark blue background
	nv.renderer.Clear()
}

// Present updates the screen
func (nv *NetworkVisualizer) Present() {
	nv.renderer.Present()
}

// RenderNetwork renders the neural network structure
func (nv *NetworkVisualizer) RenderNetwork(nn *neuralnetwork.NeuralNetwork, activations [][]float64) {
	nv.Clear()

	// Calculate layer positions
	layerSpacing := float64(nv.width) / float64(len(nn.HiddenLayers)+2)
	nodeRadius := int32(8)

	// Draw input layer
	inputX := layerSpacing / 2
	inputYSpacing := float64(nv.height) / float64(nn.NumInputs+1)
	for i := 0; i < nn.NumInputs; i++ {
		y := inputYSpacing * float64(i+1)
		nv.drawNode(int32(inputX), int32(y), nodeRadius, 0.5) // Input nodes are neutral
	}

	// Draw hidden layers
	for layerIdx, layerSize := range nn.HiddenLayers {
		x := layerSpacing * float64(layerIdx+1) + layerSpacing/2
		ySpacing := float64(nv.height) / float64(layerSize+1)

		for nodeIdx := 0; nodeIdx < layerSize; nodeIdx++ {
			y := ySpacing * float64(nodeIdx+1)
			activation := 0.0
			if activations != nil && layerIdx < len(activations) && nodeIdx < len(activations[layerIdx]) {
				activation = activations[layerIdx][nodeIdx]
			}
			nv.drawNode(int32(x), int32(y), nodeRadius, activation)
		}
	}

	// Draw output layer
	outputX := layerSpacing * float64(len(nn.HiddenLayers)+1) + layerSpacing/2
	outputYSpacing := float64(nv.height) / float64(nn.NumOutputs+1)
	for i := 0; i < nn.NumOutputs; i++ {
		y := outputYSpacing * float64(i+1)
		activation := 0.0
		if activations != nil && len(activations) > len(nn.HiddenLayers) && i < len(activations[len(nn.HiddenLayers)]) {
			activation = activations[len(nn.HiddenLayers)][i]
		}
		nv.drawNode(int32(outputX), int32(y), nodeRadius, activation)
	}

	// Draw connections
	nv.drawConnections(nn)

	nv.Present()
}

// drawNode draws a single neuron node with color based on activation
func (nv *NetworkVisualizer) drawNode(x, y, radius int32, activation float64) {
	// Color based on activation (blue for negative, red for positive)
	intensity := math.Abs(activation)
	if intensity > 1.0 {
		intensity = 1.0
	}

	var r, g, b uint8
	if activation >= 0 {
		r = uint8(255 * intensity)
		g = uint8(100 * intensity)
		b = uint8(100 * intensity)
	} else {
		r = uint8(100 * intensity)
		g = uint8(100 * intensity)
		b = uint8(255 * intensity)
	}

	nv.renderer.SetDrawColor(r, g, b, 255)
	nv.renderer.FillRect(&sdl.Rect{X: x - radius, Y: y - radius, W: radius * 2, H: radius * 2})

	// Draw border
	nv.renderer.SetDrawColor(255, 255, 255, 255)
	nv.renderer.DrawRect(&sdl.Rect{X: x - radius, Y: y - radius, W: radius * 2, H: radius * 2})
}

// drawConnections draws the weighted connections between layers
func (nv *NetworkVisualizer) drawConnections(nn *neuralnetwork.NeuralNetwork) {
	layerSpacing := float64(nv.width) / float64(len(nn.HiddenLayers)+2)

	// Connections from input to first hidden layer
	inputX := layerSpacing / 2
	inputYSpacing := float64(nv.height) / float64(nn.NumInputs+1)
	hiddenX := layerSpacing * 1 + layerSpacing/2
	hiddenYSpacing := float64(nv.height) / float64(nn.HiddenLayers[0]+1)

	for i := 0; i < nn.NumInputs; i++ {
		inputY := inputYSpacing * float64(i+1)
		for j := 0; j < nn.HiddenLayers[0]; j++ {
			hiddenY := hiddenYSpacing * float64(j+1)
			weight := nn.HiddenWeights[0][j][i]
			nv.drawConnection(int32(inputX), int32(inputY), int32(hiddenX), int32(hiddenY), weight)
		}
	}

	// Connections between hidden layers
	for layerIdx := 0; layerIdx < len(nn.HiddenLayers)-1; layerIdx++ {
		fromX := layerSpacing * float64(layerIdx+1) + layerSpacing/2
		fromYSpacing := float64(nv.height) / float64(nn.HiddenLayers[layerIdx]+1)
		toX := layerSpacing * float64(layerIdx+2) + layerSpacing/2
		toYSpacing := float64(nv.height) / float64(nn.HiddenLayers[layerIdx+1]+1)

		for fromNode := 0; fromNode < nn.HiddenLayers[layerIdx]; fromNode++ {
			fromY := fromYSpacing * float64(fromNode+1)
			for toNode := 0; toNode < nn.HiddenLayers[layerIdx+1]; toNode++ {
				toY := toYSpacing * float64(toNode+1)
				weight := nn.HiddenWeights[layerIdx+1][toNode][fromNode]
				nv.drawConnection(int32(fromX), int32(fromY), int32(toX), int32(toY), weight)
			}
		}
	}

	// Connections from last hidden layer to output
	if len(nn.HiddenLayers) > 0 {
		hiddenX = layerSpacing * float64(len(nn.HiddenLayers)) + layerSpacing/2
		hiddenYSpacing = float64(nv.height) / float64(nn.HiddenLayers[len(nn.HiddenLayers)-1]+1)
		outputX := layerSpacing * float64(len(nn.HiddenLayers)+1) + layerSpacing/2
		outputYSpacing := float64(nv.height) / float64(nn.NumOutputs+1)

		for i := 0; i < nn.HiddenLayers[len(nn.HiddenLayers)-1]; i++ {
			hiddenY := hiddenYSpacing * float64(i+1)
			for j := 0; j < nn.NumOutputs; j++ {
				outputY := outputYSpacing * float64(j+1)
				weight := nn.OutputWeights[j][i]
				nv.drawConnection(int32(hiddenX), int32(hiddenY), int32(outputX), int32(outputY), weight)
			}
		}
	}
}

// drawConnection draws a connection line with thickness based on weight
func (nv *NetworkVisualizer) drawConnection(x1, y1, x2, y2 int32, weight float64) {
	// Color based on weight (blue for negative, red for positive)
	intensity := math.Abs(weight)
	if intensity > 1.0 {
		intensity = 1.0
	}

	var r, g, b uint8
	if weight >= 0 {
		r = uint8(255 * intensity)
		g = uint8(150 * intensity)
		b = uint8(150 * intensity)
	} else {
		r = uint8(150 * intensity)
		g = uint8(150 * intensity)
		b = uint8(255 * intensity)
	}

	nv.renderer.SetDrawColor(r, g, b, uint8(100+155*intensity)) // Alpha based on weight strength

	// Draw line (simplified - could be improved with thicker lines)
	nv.renderer.DrawLine(x1, y1, x2, y2)
}</content>
<parameter name="filePath">internal/visualization/visualization.go