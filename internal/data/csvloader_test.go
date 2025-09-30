package data_test

import (
	"os"
	"reflect"
	"sort"
	"testing"

	"go-neuralnetwork/internal/data"
	"go-neuralnetwork/internal/tempfile"
)

func TestLoadCSV(t *testing.T) {
	// Test case 1: Valid CSV with header
	csvContent1 := `header1,header2,header3
1.0,2.0,3.0
4.0,5.0,6.0`
	filePath1, err := tempfile.CreateTempFileWithContent("testdata-*.csv", csvContent1)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath1)

	dataset, err := data.LoadCSV(filePath1, 1.0)
	if err != nil {
		t.Fatalf("Test Case 1: Unexpected error: %v", err)
	}

	// The LoadCSV function shuffles the data, so we need to sort it to have a deterministic test.
	type labeledData struct {
		input  []float64
		target []float64
	}

	var combined []labeledData
	for i := range dataset.TrainInputs {
		combined = append(combined, labeledData{input: dataset.TrainInputs[i], target: dataset.TrainTargets[i]})
	}

	// Sort based on the first element of the input slice.
	sort.Slice(combined, func(i, j int) bool {
		if len(combined[i].input) == 0 || len(combined[j].input) == 0 {
			return false
		}
		return combined[i].input[0] < combined[j].input[0]
	})

	sortedInputs := make([][]float64, len(combined))
	sortedTargets := make([][]float64, len(combined))
	for i, d := range combined {
		sortedInputs[i] = d.input
		sortedTargets[i] = d.target
	}

	expectedInputs1 := [][]float64{{0.0, 0.0}, {1.0, 1.0}}
	expectedTargets1 := [][]float64{{0.0}, {1.0}}
	expectedTargetMins1 := []float64{3.0}
	expectedTargetMaxs1 := []float64{6.0}
	expectedInputMins1 := []float64{1.0, 2.0}
	expectedInputMaxs1 := []float64{4.0, 5.0}

	if !reflect.DeepEqual(sortedInputs, expectedInputs1) {
		t.Errorf("Test Case 1: Inputs mismatch. Got %v, Expected %v", sortedInputs, expectedInputs1)
	}
	if !reflect.DeepEqual(sortedTargets, expectedTargets1) {
		t.Errorf("Test Case 1: Targets mismatch. Got %v, Expected %v", sortedTargets, expectedTargets1)
	}
	if !reflect.DeepEqual(dataset.TargetMins, expectedTargetMins1) {
		t.Errorf("Test Case 1: TargetMins mismatch. Got %v, Expected %v", dataset.TargetMins, expectedTargetMins1)
	}
	if !reflect.DeepEqual(dataset.TargetMaxs, expectedTargetMaxs1) {
		t.Errorf("Test Case 1: TargetMaxs mismatch. Got %v, Expected %v", dataset.TargetMaxs, expectedTargetMaxs1)
	}
	if !reflect.DeepEqual(dataset.InputMins, expectedInputMins1) {
		t.Errorf("Test Case 1: InputMins mismatch. Got %v, Expected %v", dataset.InputMins, expectedInputMins1)
	}
	if !reflect.DeepEqual(dataset.InputMaxs, expectedInputMaxs1) {
		t.Errorf("Test Case 1: InputMaxs mismatch. Got %v, Expected %v", dataset.InputMaxs, expectedInputMaxs1)
	}
	if len(dataset.TestInputs) != 0 {
		t.Errorf("Test Case 1: TestInputs should be empty with splitRatio 1.0, got %v", dataset.TestInputs)
	}
	if len(dataset.TestTargets) != 0 {
		t.Errorf("Test Case 1: TestTargets should be empty with splitRatio 1.0, got %v", dataset.TestTargets)
	}

	// Test case 2: Invalid file path
	_, err = data.LoadCSV("nonexistent.csv", 1.0)
	if err == nil {
		t.Errorf("Test Case 2: Expected an error for invalid file path, got nil")
	}

	// Test case 3: Empty CSV file
	csvContent3 := ``
	filePath3, err := tempfile.CreateTempFileWithContent("testdata-*.csv", csvContent3)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath3)

	_, err = data.LoadCSV(filePath3, 1.0)
	if err == nil {
		t.Errorf("Test Case 3: Expected an error for empty CSV, got nil")
	}

	// Test case 4: CSV with non-numeric data
	csvContent4 := `header1,header2
abc,1.0`
	filePath4, err := tempfile.CreateTempFileWithContent("testdata-*.csv", csvContent4)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath4)

	_, err = data.LoadCSV(filePath4, 1.0)
	if err == nil {
		t.Errorf("Test Case 4: Expected an error for non-numeric data, got nil")
	}
}

func TestLoadCSVForClassification(t *testing.T) {
	// Test case 1: Valid classification CSV
	csvContent1 := `sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor`
	filePath1, err := tempfile.CreateTempFileWithContent("classification-*.csv", csvContent1)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath1)

	dataset, err := data.LoadCSVForClassification(filePath1, 1.0)
	if err != nil {
		t.Fatalf("Test Case 1: Unexpected error: %v", err)
	}

	// The LoadCSVForClassification function shuffles the data, so we need to sort it to have a deterministic test.
	type labeledData struct {
		input  []float64
		target []float64
	}

	var combined []labeledData
	for i := range dataset.TrainInputs {
		combined = append(combined, labeledData{input: dataset.TrainInputs[i], target: dataset.TrainTargets[i]})
	}

	// Sort based on the first element of the input slice.
	sort.Slice(combined, func(i, j int) bool {
		if len(combined[i].input) == 0 || len(combined[j].input) == 0 {
			return false
		}
		return combined[i].input[0] < combined[j].input[0]
	})

	sortedInputs := make([][]float64, len(combined))
	sortedTargets := make([][]float64, len(combined))
	for i, d := range combined {
		sortedInputs[i] = d.input
		sortedTargets[i] = d.target
	}

	// Note: The expected inputs are normalized.
	expectedInputs1 := [][]float64{{0.0, 1.0, 0.0, 0.0}, {1.0, 0.0, 1.0, 1.0}}
	expectedTargets1 := [][]float64{{1.0, 0.0}, {0.0, 1.0}}
	expectedClassMap1 := map[string]int{"setosa": 0, "versicolor": 1}

	if !reflect.DeepEqual(sortedInputs, expectedInputs1) {
		t.Errorf("Test Case 1: Inputs mismatch. Got %v, Expected %v", sortedInputs, expectedInputs1)
	}
	if !reflect.DeepEqual(sortedTargets, expectedTargets1) {
		t.Errorf("Test Case 1: Targets mismatch. Got %v, Expected %v", sortedTargets, expectedTargets1)
	}
	if !reflect.DeepEqual(dataset.ClassMap, expectedClassMap1) {
		t.Errorf("Test Case 1: ClassMap mismatch. Got %v, Expected %v", dataset.ClassMap, expectedClassMap1)
	}

	// Test case 2: CSV with non-numeric input data
	csvContent2 := `header1,header2
abc,classA`
	filePath2, err := tempfile.CreateTempFileWithContent("classification-*.csv", csvContent2)
	if err != nil {
		t.Fatalf("Failed to create temp CSV: %v", err)
	}
	defer os.Remove(filePath2)

	_, err = data.LoadCSVForClassification(filePath2, 1.0)
	if err == nil {
		t.Errorf("Test Case 2: Expected an error for non-numeric data, got nil")
	}
}

func TestSplitData(t *testing.T) {
	inputs := [][]float64{{1}, {2}, {3}, {4}, {5}}
	targets := [][]float64{{10}, {20}, {30}, {40}, {50}}

	// Test case 1: 80/20 split
	trainInputs, trainTargets, testInputs, testTargets := data.SplitData(inputs, targets, 0.8)

	if len(trainInputs) != 4 || len(trainTargets) != 4 {
		t.Errorf("Expected 4 training items, got %d inputs and %d targets", len(trainInputs), len(trainTargets))
	}
	if len(testInputs) != 1 || len(testTargets) != 1 {
		t.Errorf("Expected 1 test item, got %d inputs and %d targets", len(testInputs), len(testTargets))
	}
	if !reflect.DeepEqual(trainInputs[3], []float64{4}) {
		t.Errorf("Last training input is incorrect")
	}
	if !reflect.DeepEqual(testInputs[0], []float64{5}) {
		t.Errorf("First test input is incorrect")
	}

	// Test case 2: 100% split (all training)
	trainInputs, _, testInputs, _ = data.SplitData(inputs, targets, 1.0)
	if len(trainInputs) != 5 {
		t.Errorf("Expected 5 training inputs, got %d", len(trainInputs))
	}
	if len(testInputs) != 0 {
		t.Errorf("Expected 0 test inputs, got %d", len(testInputs))
	}

	// Test case 3: 0% split (all testing)
	trainInputs, _, testInputs, _ = data.SplitData(inputs, targets, 0.0)
	if len(trainInputs) != 0 {
		t.Errorf("Expected 0 training inputs, got %d", len(trainInputs))
	}
	if len(testInputs) != 5 {
		t.Errorf("Expected 5 test inputs, got %d", len(testInputs))
	}

	// Test case 4: Empty input
	trainInputs, trainTargets, testInputs, testTargets = data.SplitData([][]float64{}, [][]float64{}, 0.5)
	if len(trainInputs) != 0 || len(trainTargets) != 0 || len(testInputs) != 0 || len(testTargets) != 0 {
		t.Errorf("Expected empty slices for empty input, got non-empty slices")
	}
}

func TestShuffle(t *testing.T) {
	inputs := [][]float64{{1}, {2}, {3}, {4}, {5}}
	targets := [][]float64{{10}, {20}, {30}, {40}, {50}}

	// Create a map to check if input-target pairs are preserved
	originalPairs := make(map[float64]float64)
	for i := range inputs {
		originalPairs[inputs[i][0]] = targets[i][0]
	}

	// Create copies to shuffle
	shuffledInputs := make([][]float64, len(inputs))
	shuffledTargets := make([][]float64, len(targets))
	for i := range inputs {
		shuffledInputs[i] = make([]float64, len(inputs[i]))
		copy(shuffledInputs[i], inputs[i])
	}
	for i := range targets {
		shuffledTargets[i] = make([]float64, len(targets[i]))
		copy(shuffledTargets[i], targets[i])
	}

	data.Shuffle(shuffledInputs, shuffledTargets)

	if len(shuffledInputs) != len(inputs) {
		t.Errorf("Shuffle changed the length of inputs slice. Got %d, expected %d", len(shuffledInputs), len(inputs))
	}
	if len(shuffledTargets) != len(targets) {
		t.Errorf("Shuffle changed the length of targets slice. Got %d, expected %d", len(shuffledTargets), len(targets))
	}

	// Check that pairs are preserved
	for i := range shuffledInputs {
		inputVal := shuffledInputs[i][0]
		targetVal := shuffledTargets[i][0]
		if expectedTarget, ok := originalPairs[inputVal]; !ok || expectedTarget != targetVal {
			t.Errorf("Shuffle broke input-target pair. For input %v, expected target %v, but got %v", inputVal, originalPairs[inputVal], targetVal)
		}
	}

	// Check that the data is actually shuffled (this test might fail by chance, but it's unlikely with a small set)
	if reflect.DeepEqual(inputs, shuffledInputs) {
		t.Log("Warning: Data was not shuffled. This might happen by chance, re-run test.")
	}
}
