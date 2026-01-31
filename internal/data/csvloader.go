package data

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"
)

type Dataset struct {
	TrainInputs  [][]float64
	TrainTargets [][]float64
	TestInputs   [][]float64
	TestTargets  [][]float64
	InputSize    int
	OutputSize   int
	InputMins    []float64
	InputMaxs    []float64
	TargetMins   []float64
	TargetMaxs   []float64
	ClassMap     map[string]int
}

func Shuffle(inputs, targets [][]float64) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	r.Shuffle(len(inputs), func(i, j int) {
		inputs[i], inputs[j] = inputs[j], inputs[i]
		targets[i], targets[j] = targets[j], targets[i]
	})
}

func SplitData(inputs, targets [][]float64, splitRatio float64) (trainInputs, trainTargets, testInputs, testTargets [][]float64) {
	splitIndex := int(float64(len(inputs)) * splitRatio)
	trainInputs = inputs[:splitIndex]
	trainTargets = targets[:splitIndex]
	testInputs = inputs[splitIndex:]
	testTargets = targets[splitIndex:]
	return
}

// findMinMax computes per-column min and max values for the given column indices in records.
func findMinMax(records [][]string, columns int) (mins, maxs []float64, err error) {
	mins = make([]float64, columns)
	maxs = make([]float64, columns)
	for i := range mins {
		mins[i] = 1e9
		maxs[i] = -1e9
	}
	for _, record := range records {
		for i := 0; i < columns; i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("error parsing float in record %v: %w", record, err)
			}
			if val < mins[i] {
				mins[i] = val
			}
			if val > maxs[i] {
				maxs[i] = val
			}
		}
	}
	return mins, maxs, nil
}

// normalizeValue applies min-max normalization, returning 0 when max == min.
func normalizeValue(val, min, max float64) float64 {
	if max-min == 0 {
		return 0
	}
	return (val - min) / (max - min)
}

func LoadCSV(filePath string, splitRatio float64) (*Dataset, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return nil, err
	}

	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) > 0 {
		if _, err := strconv.ParseFloat(records[0][len(records[0])-1], 64); err != nil {
			return loadCSVForClassification(header, records, splitRatio)
		}
	}

	inputSize := len(header) - 1
	outputSize := 1
	totalCols := inputSize + outputSize

	// Find min and max for all columns
	mins, maxs, err := findMinMax(records, totalCols)
	if err != nil {
		return nil, err
	}

	inputMins := mins[:inputSize]
	inputMaxs := maxs[:inputSize]
	targetMins := mins[inputSize:]
	targetMaxs := maxs[inputSize:]

	var inputs, targets [][]float64
	for _, record := range records {
		inputRow := make([]float64, inputSize)
		outputRow := make([]float64, outputSize)

		for i := range inputRow {
			val, _ := strconv.ParseFloat(record[i], 64)
			inputRow[i] = normalizeValue(val, mins[i], maxs[i])
		}

		for i := range outputRow {
			val, _ := strconv.ParseFloat(record[inputSize+i], 64)
			outputRow[i] = normalizeValue(val, mins[inputSize+i], maxs[inputSize+i])
		}

		inputs = append(inputs, inputRow)
		targets = append(targets, outputRow)
	}

	Shuffle(inputs, targets)
	trainInputs, trainTargets, testInputs, testTargets := SplitData(inputs, targets, splitRatio)

	return &Dataset{
		TrainInputs:  trainInputs,
		TrainTargets: trainTargets,
		TestInputs:   testInputs,
		TestTargets:  testTargets,
		InputSize:    inputSize,
		OutputSize:   outputSize,
		InputMins:    inputMins,
		InputMaxs:    inputMaxs,
		TargetMins:   targetMins,
		TargetMaxs:   targetMaxs,
	}, nil
}

func LoadCSVForClassification(filePath string, splitRatio float64) (*Dataset, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return nil, err
	}

	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return loadCSVForClassification(header, records, splitRatio)
}

func loadCSVForClassification(header []string, records [][]string, splitRatio float64) (*Dataset, error) {
	inputSize := len(header) - 1
	classMap := make(map[string]int)
	classIndex := 0

	// First pass: find all unique class names
	for _, record := range records {
		className := record[inputSize]
		if _, exists := classMap[className]; !exists {
			classMap[className] = classIndex
			classIndex++
		}
	}
	outputSize := len(classMap)

	// Find min and max for input columns to normalize the data
	inputMins, inputMaxs, err := findMinMax(records, inputSize)
	if err != nil {
		return nil, err
	}

	var inputs, targets [][]float64
	// Second pass: build the inputs and targets slices
	for _, record := range records {
		inputRow := make([]float64, inputSize)
		for i := range inputRow {
			val, _ := strconv.ParseFloat(record[i], 64)
			inputRow[i] = normalizeValue(val, inputMins[i], inputMaxs[i])
		}
		inputs = append(inputs, inputRow)

		targetRow := make([]float64, outputSize)
		className := record[inputSize]
		targetRow[classMap[className]] = 1.0 // One-hot encoding
		targets = append(targets, targetRow)
	}

	Shuffle(inputs, targets)
	trainInputs, trainTargets, testInputs, testTargets := SplitData(inputs, targets, splitRatio)

	return &Dataset{
		TrainInputs:  trainInputs,
		TrainTargets: trainTargets,
		TestInputs:   testInputs,
		TestTargets:  testTargets,
		InputSize:    inputSize,
		OutputSize:   outputSize,
		InputMins:    inputMins,
		InputMaxs:    inputMaxs,
		ClassMap:     classMap,
	}, nil
}
