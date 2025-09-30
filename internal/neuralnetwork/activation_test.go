package neuralnetwork

import (
	"math"
	"testing"
)

const floatTolerance = 1e-9

func TestSigmoid(t *testing.T) {
	sigmoid := &Sigmoid{}
	testCases := []struct {
		name             string
		input            float64
		expectedActivate float64
	}{
		{"zero", 0, 0.5},
		{"positive", 2.0, 1 / (1 + math.Exp(-2.0))},
		{"negative", -2.0, 1 / (1 + math.Exp(2.0))},
	}

	for _, tc := range testCases {
		t.Run(tc.name+"_Activate", func(t *testing.T) {
			result := sigmoid.Activate(tc.input)
			if math.Abs(result-tc.expectedActivate) > floatTolerance {
				t.Errorf("Expected Activate(%f) to be %f, but got %f", tc.input, tc.expectedActivate, result)
			}
		})
	}

	derivativeTestCases := []struct {
		name               string
		activatedInput     float64
		expectedDerivative float64
	}{
		{"from_zero", 0.5, 0.25},     // Sigmoid(0) = 0.5, Derivative = 0.5 * (1-0.5) = 0.25
		{"from_positive", 0.8, 0.16}, // Derivative = 0.8 * (1-0.8) = 0.16
	}

	for _, tc := range derivativeTestCases {
		t.Run(tc.name+"_Derivative", func(t *testing.T) {
			result := sigmoid.Derivative(tc.activatedInput)
			if math.Abs(result-tc.expectedDerivative) > floatTolerance {
				t.Errorf("Expected Derivative(%f) to be %f, but got %f", tc.activatedInput, tc.expectedDerivative, result)
			}
		})
	}
}

func TestTanh(t *testing.T) {
	tanh := &Tanh{}
	testCases := []struct {
		name             string
		input            float64
		expectedActivate float64
	}{
		{"zero", 0, 0},
		{"positive", 1, math.Tanh(1)},
		{"negative", -1, math.Tanh(-1)},
	}

	for _, tc := range testCases {
		t.Run(tc.name+"_Activate", func(t *testing.T) {
			result := tanh.Activate(tc.input)
			if math.Abs(result-tc.expectedActivate) > floatTolerance {
				t.Errorf("Expected Activate(%f) to be %f, but got %f", tc.input, tc.expectedActivate, result)
			}
		})
	}

	derivativeTestCases := []struct {
		name               string
		activatedInput     float64
		expectedDerivative float64
	}{
		{"from_zero", 0.0, 1.0},      // Tanh(0) = 0, Derivative = 1 - 0^2 = 1
		{"from_positive", 0.5, 0.75}, // Derivative = 1 - 0.5^2 = 0.75
	}

	for _, tc := range derivativeTestCases {
		t.Run(tc.name+"_Derivative", func(t *testing.T) {
			result := tanh.Derivative(tc.activatedInput)
			if math.Abs(result-tc.expectedDerivative) > floatTolerance {
				t.Errorf("Expected Derivative(%f) to be %f, but got %f", tc.activatedInput, tc.expectedDerivative, result)
			}
		})
	}
}

func TestReLU(t *testing.T) {
	relu := &ReLU{}
	testCases := []struct {
		name             string
		input            float64
		expectedActivate float64
	}{
		{"zero", 0, 0},
		{"positive", 5.5, 5.5},
		{"negative", -5.5, 0},
	}

	for _, tc := range testCases {
		t.Run(tc.name+"_Activate", func(t *testing.T) {
			result := relu.Activate(tc.input)
			if math.Abs(result-tc.expectedActivate) > floatTolerance {
				t.Errorf("Expected Activate(%f) to be %f, but got %f", tc.input, tc.expectedActivate, result)
			}
		})
	}

	derivativeTestCases := []struct {
		name               string
		activatedInput     float64
		expectedDerivative float64
	}{
		{"zero", 0, 0},
		{"positive", 5.5, 1},
		{"negative", -5.5, 0}, // Although the input to derivative is the activated output, test case remains valid
	}

	for _, tc := range derivativeTestCases {
		t.Run(tc.name+"_Derivative", func(t *testing.T) {
			result := relu.Derivative(tc.activatedInput)
			if math.Abs(result-tc.expectedDerivative) > floatTolerance {
				t.Errorf("Expected Derivative(%f) to be %f, but got %f", tc.activatedInput, tc.expectedDerivative, result)
			}
		})
	}
}

func TestLinear(t *testing.T) {
	linear := &Linear{}
	testCases := []struct {
		name             string
		input            float64
		expectedActivate float64
	}{
		{"zero", 0, 0},
		{"positive", 42.1, 42.1},
		{"negative", -10.5, -10.5},
	}

	for _, tc := range testCases {
		t.Run(tc.name+"_Activate", func(t *testing.T) {
			result := linear.Activate(tc.input)
			if math.Abs(result-tc.expectedActivate) > floatTolerance {
				t.Errorf("Expected Activate(%f) to be %f, but got %f", tc.input, tc.expectedActivate, result)
			}
		})
	}

	derivativeTestCases := []struct {
		name               string
		activatedInput     float64
		expectedDerivative float64
	}{
		{"any", 123.45, 1},
	}

	for _, tc := range derivativeTestCases {
		t.Run(tc.name+"_Derivative", func(t *testing.T) {
			result := linear.Derivative(tc.activatedInput)
			if math.Abs(result-tc.expectedDerivative) > floatTolerance {
				t.Errorf("Expected Derivative(%f) to be %f, but got %f", tc.activatedInput, tc.expectedDerivative, result)
			}
		})
	}
}

func TestGetActivation(t *testing.T) {
	t.Run("ValidActivation", func(t *testing.T) {
		activation, err := GetActivation("relu")
		if err != nil {
			t.Fatalf("Expected no error for valid activation 'relu', but got %v", err)
		}
		if _, ok := activation.(*ReLU); !ok {
			t.Errorf("Expected a ReLU activation function, but got %T", activation)
		}
	})

	t.Run("InvalidActivation", func(t *testing.T) {
		_, err := GetActivation("unknown_activation")
		if err == nil {
			t.Fatal("Expected an error for invalid activation 'unknown_activation', but got nil")
		}
	})
}
