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
			return LoadCSVForClassification(filePath, splitRatio)
		}
	}

	// Reset reader and read records again
	file.Seek(0, 0)
	reader = csv.NewReader(file)
	header, err = reader.Read()
	if err != nil {
		return nil, err
	}

	inputSize := len(header) - 1
	outputSize := 1

	// Find min and max for each column to normalize the data
	mins := make([]float64, inputSize+outputSize)
	maxs := make([]float64, inputSize+outputSize)
	for i := range mins {
		mins[i] = 1e9
		maxs[i] = -1e9
	}

	for _, record := range records {
		for i := range mins {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, err
			}
			if val < mins[i] {
				mins[i] = val
			}
			if val > maxs[i] {
				maxs[i] = val
			}
		}
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
			if maxs[i]-mins[i] == 0 {
				inputRow[i] = 0
			} else {
				inputRow[i] = (val - mins[i]) / (maxs[i] - mins[i])
			}
		}

		for i := range outputRow {
			val, _ := strconv.ParseFloat(record[inputSize+i], 64)
			if maxs[inputSize+i]-mins[inputSize+i] == 0 {
				outputRow[i] = 0
			} else {
				outputRow[i] = (val - mins[inputSize+i]) / (maxs[inputSize+i] - mins[inputSize+i])
			}
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
	inputMins := make([]float64, inputSize)
	inputMaxs := make([]float64, inputSize)
	for i := range inputMins {
		inputMins[i] = 1e9
		inputMaxs[i] = -1e9
	}

	for _, record := range records {
		for i := range inputMins {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, fmt.Errorf("error parsing float in record %v: %w", record, err)
			}
			if val < inputMins[i] {
				inputMins[i] = val
			}
			if val > inputMaxs[i] {
				inputMaxs[i] = val
			}
		}
	}

	var inputs, targets [][]float64
	// Second pass: build the inputs and targets slices
	for _, record := range records {
		inputRow := make([]float64, inputSize)
		for i := range inputRow {
			val, _ := strconv.ParseFloat(record[i], 64)
			if inputMaxs[i]-inputMins[i] == 0 {
				inputRow[i] = 0
			} else {
				inputRow[i] = (val - inputMins[i]) / (inputMaxs[i] - inputMins[i])
			}
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
