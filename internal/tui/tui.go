package tui

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"go-neuralnetwork/internal/data"
	"go-neuralnetwork/internal/neuralnetwork"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Messages
type (
	csvFilesLoadedMsg  struct{ files []string }
	modelsLoadedMsg    struct{ models []string }
	trainingStartedMsg struct{}
	epochCompletedMsg  struct {
		epochNum int
		loss     float64
	}
	trainingFinishedMsg struct {
		modelData *data.ModelData
		testData  *data.Dataset
	}
	evaluationFinishedMsg             struct{ accuracy float64 }
	predictionResultMsg               struct{ result float64 }
	predictionResultClassificationMsg struct{ result string }
	errorMsg                          struct{ err error }
)

func (m *Model) runTraining() tea.Cmd {
	return func() tea.Msg {
		// --- Input Validation ---
		csvIndex, err := strconv.Atoi(m.trainingForm.inputs[0].Value())
		if err != nil || csvIndex < 1 || csvIndex > len(m.trainingForm.csvFiles) {
			return errorMsg{fmt.Errorf("invalid CSV file selection")}
		}
		csvPath := m.trainingForm.csvFiles[csvIndex-1]
		layersStr := m.trainingForm.inputs[1].Value()
		if layersStr == "" {
			layersStr = "20,20"
		}
		hiddenLayers := []int{}
		for _, s := range strings.Split(layersStr, ",") {
			i, err := strconv.Atoi(strings.TrimSpace(s))
			if err != nil {
				return errorMsg{fmt.Errorf("invalid hidden layers: %w", err)}
			}
			hiddenLayers = append(hiddenLayers, i)
		}
		activationsStr := m.trainingForm.inputs[2].Value()
		if activationsStr == "" {
			activationsStr = "relu,relu"
		}
		hiddenActivations := strings.Split(activationsStr, ",")
		outputActivation := m.trainingForm.inputs[3].Value()
		if outputActivation == "" {
			outputActivation = "linear"
		}
		epochsStr := m.trainingForm.inputs[4].Value()
		if epochsStr == "" {
			epochsStr = "1000"
		}
		epochs, err := strconv.Atoi(epochsStr)
		if err != nil {
			return errorMsg{fmt.Errorf("invalid epochs value: %w", err)}
		}
		lrStr := m.trainingForm.inputs[5].Value()
		if lrStr == "" {
			lrStr = "0.001"
		}
		learningRate, err := strconv.ParseFloat(lrStr, 64)
		if err != nil {
			return errorMsg{fmt.Errorf("invalid learning rate: %w", err)}
		}
		egStr := m.trainingForm.inputs[6].Value()
		if egStr == "" {
			egStr = "0.001"
		}
		errorGoal, err := strconv.ParseFloat(egStr, 64)
		if err != nil {
			return errorMsg{fmt.Errorf("invalid error goal: %w", err)}
		}

		// Load data
		dataset, err := data.LoadCSV(csvPath, 0.8)
		if err != nil {
			return errorMsg{fmt.Errorf("failed to load CSV data: %w", err)}
		}

		// Initialize network
		nn := neuralnetwork.InitNetwork(dataset.InputSize, hiddenLayers, dataset.OutputSize, hiddenActivations, outputActivation)

		// This channel will receive training progress
		progressChan := make(chan any)

		// Goroutine to run training and send messages
		go func() {
			nn.Train(dataset.TrainInputs, dataset.TrainTargets, epochs, learningRate, errorGoal, progressChan)
			modelData := &data.ModelData{
				NN:         nn,
				InputMins:  dataset.InputMins,
				InputMaxs:  dataset.InputMaxs,
				TargetMins: dataset.TargetMins,
				TargetMaxs: dataset.TargetMaxs,
				ClassMap:   dataset.ClassMap,
			}
			m.program.Send(trainingFinishedMsg{modelData: modelData, testData: dataset})
		}()

		// Goroutine to listen for progress and update the TUI
		go func() {
			epochNum := 1
			for loss := range progressChan {
				m.program.Send(epochCompletedMsg{epochNum: epochNum, loss: loss.(float64)})
				epochNum++
			}
		}()

		return trainingStartedMsg{}
	}
}

func findCsvFiles() tea.Msg {
	files, err := filepath.Glob("*.csv")
	if err != nil {
		return errorMsg{err}
	}
	return csvFilesLoadedMsg{files}
}

func findModels() tea.Msg {
	files, err := filepath.Glob("saved_models/*.json")
	if err != nil {
		return errorMsg{err}
	}
	return modelsLoadedMsg{models: files}
}

// sessionState represents the current view of the TUI.
type sessionState uint

const (
	mainMenu sessionState = iota
	trainingForm
	trainingInProgress
	evaluation
	predictionForm
	predictionResult
	saveModelForm
	errorView
)

// Styles
var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FAFAFA")).
			Background(lipgloss.Color("#7D56F4")).
			PaddingLeft(1).
			PaddingRight(1)

	menuItemStyle         = lipgloss.NewStyle().PaddingLeft(2)
	selectedMenuItemStyle = lipgloss.NewStyle().PaddingLeft(2).Foreground(lipgloss.Color("205"))
	helpStyle             = lipgloss.NewStyle().Foreground(lipgloss.Color("241")).Padding(1, 0, 0, 2)
	focusedStyle          = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))
	errorStyle            = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF8700"))
)

// Model represents the state of the entire application.
type Model struct {
	state           sessionState
	menuCursor      int
	menuChoices     []string
	trainingForm    trainingFormModel
	predictionForm  predictionFormModel
	saveModelInput  textinput.Model
	modelData       *data.ModelData
	program         *tea.Program
	lastError       error
	quitting        bool
	terminalWidth   int
	terminalHeight  int
	lastLoss        float64
	currentEpoch    int
	totalEpochs     int
	predictionValue float64
	predictionClass string
	accuracy        float64
}

// trainingFormModel holds the state for the training configuration form.
type trainingFormModel struct {
	focusIndex int
	inputs     []textinput.Model
	csvFiles   []string
}

// predictionFormModel holds the state for the prediction form.
type predictionFormModel struct {
	focusIndex int
	inputs     []textinput.Model
	models     []string
}

func newTrainingForm() trainingFormModel {
	m := trainingFormModel{
		inputs: make([]textinput.Model, 7),
	}

	var t textinput.Model
	for i := range m.inputs {
		t = textinput.New()
		t.Cursor.Style = focusedStyle
		t.CharLimit = 32

		switch i {
		case 0:
			t.Placeholder = "1"
			t.Focus()
		case 1:
			t.Placeholder = "20,20"
		case 2:
			t.Placeholder = "relu,relu"
		case 3:
			t.Placeholder = "linear"
		case 4:
			t.Placeholder = "1000"
		case 5:
			t.Placeholder = "0.001"
		case 6:
			t.Placeholder = "0.001"
		}
		m.inputs[i] = t
	}

	return m
}

func newPredictionForm() predictionFormModel {
	m := predictionFormModel{
		inputs: make([]textinput.Model, 2),
	}

	var t textinput.Model
	for i := range m.inputs {
		t = textinput.New()
		t.Cursor.Style = focusedStyle
		t.CharLimit = 128

		switch i {
		case 0:
			t.Placeholder = "1"
			t.Focus()
		case 1:
			t.Placeholder = "7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4"
		}
		m.inputs[i] = t
	}

	return m
}

// New creates a new TUI model.
func New() *Model {
	saveInput := textinput.New()
	saveInput.Placeholder = "my-awesome-model"
	saveInput.Focus()
	saveInput.CharLimit = 64
	saveInput.Width = 50

	return &Model{
		state:          mainMenu,
		menuChoices:    []string{"Train New Model", "Load Model & Predict", "Quit"},
		trainingForm:   newTrainingForm(),
		predictionForm: newPredictionForm(),
		saveModelInput: saveInput,
	}
}

// Init initializes the TUI.
func (m *Model) Init() tea.Cmd {
	return textinput.Blink
}

// Update handles messages and updates the model.
func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.terminalWidth = msg.Width
		m.terminalHeight = msg.Height

	case csvFilesLoadedMsg:
		m.trainingForm.csvFiles = msg.files
		return m, nil

	case modelsLoadedMsg:
		m.predictionForm.models = msg.models
		return m, nil

	case errorMsg:
		m.lastError = msg.err
		m.state = errorView
		return m, nil

	case trainingStartedMsg:
		m.state = trainingInProgress
		epochs, _ := strconv.Atoi(m.trainingForm.inputs[4].Value())
		if epochs == 0 {
			epochs = 1000
		}
		m.totalEpochs = epochs
		return m, nil

	case epochCompletedMsg:
		m.currentEpoch = msg.epochNum
		m.lastLoss = msg.loss
		return m, nil

	case trainingFinishedMsg:
		m.modelData = msg.modelData
		m.state = evaluation
		return m, func() tea.Msg {
			correct := 0
			for i, input := range msg.testData.TestInputs {
				_, prediction := m.modelData.NN.FeedForward(input)
				if msg.testData.ClassMap != nil {
					max := -1.0
					maxIndex := -1
					for i, val := range prediction {
						if val > max {
							max = val
							maxIndex = i
						}
					}

					actualIndex := -1
					for i, val := range msg.testData.TestTargets[i] {
						if val == 1.0 {
							actualIndex = i
							break
						}
					}

					if maxIndex == actualIndex {
						correct++
					}
				} else {
					// This is a regression problem, so we can't calculate accuracy.
					// We could calculate loss here instead.
				}
			}
			accuracy := float64(correct) / float64(len(msg.testData.TestInputs))
			return evaluationFinishedMsg{accuracy: accuracy}
		}

	case evaluationFinishedMsg:
		m.accuracy = msg.accuracy
		m.state = saveModelForm
		return m, nil

	case predictionResultMsg:
		m.state = predictionResult
		m.predictionValue = msg.result
		return m, nil

	case predictionResultClassificationMsg:
		m.state = predictionResult
		m.predictionClass = msg.result
		return m, nil

	case tea.KeyMsg:
		switch m.state {
		case mainMenu:
			return m.updateMainMenu(msg)
		case trainingForm:
			return m.updateTrainingForm(msg)
		case trainingInProgress:
			if msg.String() == "q" {
				m.state = mainMenu
			}
			return m, nil
		case evaluation:
			if msg.String() == "enter" || msg.String() == "q" {
				m.state = saveModelForm
			}
			return m, nil
		case predictionForm:
			return m.updatePredictionForm(msg)
		case predictionResult:
			if msg.String() == "enter" || msg.String() == "q" {
				m.state = mainMenu
			}
			return m, nil
		case saveModelForm:
			return m.updateSaveModelForm(msg)
		case errorView:
			if msg.String() == "enter" || msg.String() == "q" {
				m.state = mainMenu
			}
			return m, nil
		}
	}

	// Handle character input for the forms
	if m.state == trainingForm {
		cmd := m.updateTrainingInputs(msg)
		return m, cmd
	}
	if m.state == predictionForm {
		cmd := m.updatePredictionInputs(msg)
		return m, cmd
	}

	return m, nil
}

func (m *Model) updateMainMenu(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c", "q":
		m.quitting = true
		return m, tea.Quit

	case "up", "k":
		if m.menuCursor > 0 {
			m.menuCursor--
		}

	case "down", "j":
		if m.menuCursor < len(m.menuChoices)-1 {
			m.menuCursor++
		}

	case "enter":
		switch m.menuCursor {
		case 0:
			// Transition to training form
			m.state = trainingForm
			return m, findCsvFiles
		case 1:
			// Transition to prediction form
			m.state = predictionForm
			return m, findModels
		case 2:
			m.quitting = true
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m *Model) viewMainMenu() string {
	s := titleStyle.Render("Go Neural Network")
	s += "\n\n"

	for i, choice := range m.menuChoices {
		if m.menuCursor == i {
			s += selectedMenuItemStyle.Render(fmt.Sprintf("> %s", choice))
		} else {
			s += menuItemStyle.Render(fmt.Sprintf("  %s", choice))
		}
		s += "\n"
	}

	s += helpStyle.Render("Use arrow keys to navigate, 'enter' to select, 'q' to quit.")
	return s
}

// View renders the UI.
func (m *Model) View() string {
	if m.quitting {
		return "Quitting...\n"
	}

	var s string
	switch m.state {
	case mainMenu:
		s = m.viewMainMenu()
	case trainingForm:
		s = m.viewTrainingForm()
	case trainingInProgress:
		s = m.viewTrainingInProgress()
	case evaluation:
		s = m.viewEvaluation()
	case predictionForm:
		s = m.viewPredictionForm()
	case predictionResult:
		s = m.viewPredictionResult()
	case saveModelForm:
		s = m.viewSaveModelForm()
	case errorView:
		s = m.viewError()
	default:
		s = "Unknown state."
	}

	return s
}

func (m *Model) viewEvaluation() string {
	return fmt.Sprintf("Evaluation complete!\n\nAccuracy: %.2f%%\n\n(Press enter to continue)", m.accuracy*100)
}

func (m *Model) updateTrainingForm(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c", "q":
		// Go back to the main menu
		m.state = mainMenu
		return m, nil

	case "tab", "shift+tab", "enter", "up", "down":
		s := msg.String()

		// Did the user press enter while the button is focused?
		if s == "enter" && m.trainingForm.focusIndex == len(m.trainingForm.inputs) {
			return m, m.runTraining()
		}

		// Cycle focus
		if s == "up" || s == "shift+tab" {
			m.trainingForm.focusIndex--
		} else {
			m.trainingForm.focusIndex++
		}

		if m.trainingForm.focusIndex > len(m.trainingForm.inputs) {
			m.trainingForm.focusIndex = 0
		} else if m.trainingForm.focusIndex < 0 {
			m.trainingForm.focusIndex = len(m.trainingForm.inputs)
		}

		cmds := make([]tea.Cmd, len(m.trainingForm.inputs))
		for i := range m.trainingForm.inputs {
			if i == m.trainingForm.focusIndex {
				// Set focused state
				cmds[i] = m.trainingForm.inputs[i].Focus()
				m.trainingForm.inputs[i].PromptStyle = focusedStyle
				m.trainingForm.inputs[i].TextStyle = focusedStyle
				continue
			}
			// Remove focused state
			m.trainingForm.inputs[i].Blur()
			m.trainingForm.inputs[i].PromptStyle = lipgloss.NewStyle()
			m.trainingForm.inputs[i].TextStyle = lipgloss.NewStyle()
		}

		return m, tea.Batch(cmds...)
	}

	// Handle character input
	cmd := m.updateTrainingInputs(msg)
	return m, cmd
}

func (m *Model) updateTrainingInputs(msg tea.Msg) tea.Cmd {
	cmds := make([]tea.Cmd, len(m.trainingForm.inputs))

	// Only update the focused input
	for i := range m.trainingForm.inputs {
		if m.trainingForm.inputs[i].Focused() {
			m.trainingForm.inputs[i], cmds[i] = m.trainingForm.inputs[i].Update(msg)
		}
	}

	return tea.Batch(cmds...)
}

func (m *Model) viewTrainingForm() string {
	var b strings.Builder

	b.WriteString("Neural Network Training Configuration\n\n")

	// Render CSV file list
	b.WriteString("Available CSV Files:\n")
	if len(m.trainingForm.csvFiles) == 0 {
		b.WriteString("  (No CSV files found in current directory)\n")
	} else {
		for i, file := range m.trainingForm.csvFiles {
			b.WriteString(fmt.Sprintf("  %d: %s\n", i+1, file))
		}
	}
	b.WriteString("\n")

	// Render form
	fmt.Fprintf(&b, "Select CSV File (number): %s\n", m.trainingForm.inputs[0].View())
	fmt.Fprintf(&b, "Hidden Layers (e.g., 20,20): %s\n", m.trainingForm.inputs[1].View())

	// Activation function hints
	availableActivations := neuralnetwork.GetAvailableActivations()
	b.WriteString(fmt.Sprintf("\nAvailable activation functions: %s\n", lipgloss.NewStyle().Foreground(lipgloss.Color("241")).Render(strings.Join(availableActivations, ", "))))
	b.WriteString(fmt.Sprintf("%s\n", lipgloss.NewStyle().Foreground(lipgloss.Color("241")).Render("Hint: 'relu' or 'tanh' are common choices for hidden layers.")))
	fmt.Fprintf(&b, "Hidden Activations (e.g., relu,relu): %s\n", m.trainingForm.inputs[2].View())

	b.WriteString(fmt.Sprintf("\n%s\n", lipgloss.NewStyle().Foreground(lipgloss.Color("241")).Render("Hint: 'linear' for regression, 'sigmoid' for classification.")))
	fmt.Fprintf(&b, "Output Activation: %s\n\n", m.trainingForm.inputs[3].View())

	fmt.Fprintf(&b, "Epochs: %s\n", m.trainingForm.inputs[4].View())
	fmt.Fprintf(&b, "Learning Rate: %s\n", m.trainingForm.inputs[5].View())
	fmt.Fprintf(&b, "Error Goal: %s\n", m.trainingForm.inputs[6].View())
	b.WriteString("\n")

	// Render button
	button := "[ Start Training ]"
	if m.trainingForm.focusIndex == len(m.trainingForm.inputs) {
		b.WriteString(focusedStyle.Render(button))
	} else {
		b.WriteString(button)
	}

	b.WriteString(helpStyle.Render("\n\n  ↑/↓, tab/shift+tab: navigate | enter: select | q: back\n"))

	return b.String()
}

func (m *Model) viewTrainingInProgress() string {
	return fmt.Sprintf("Training in progress...\n\nEpoch: %d/%d\nLoss: %f\n\n(Press 'q' to stop)", m.currentEpoch, m.totalEpochs, m.lastLoss)
}

func (m *Model) updatePredictionForm(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c", "q":
		m.state = mainMenu
		return m, nil

	case "tab", "shift+tab", "enter", "up", "down":
		s := msg.String()

		if s == "enter" && m.predictionForm.focusIndex == len(m.predictionForm.inputs) {
			return m, m.runPrediction()
		}

		if s == "up" || s == "shift+tab" {
			m.predictionForm.focusIndex--
		} else {
			m.predictionForm.focusIndex++
		}

		if m.predictionForm.focusIndex > len(m.predictionForm.inputs) {
			m.predictionForm.focusIndex = 0
		} else if m.predictionForm.focusIndex < 0 {
			m.predictionForm.focusIndex = len(m.predictionForm.inputs)
		}

		cmds := make([]tea.Cmd, len(m.predictionForm.inputs))
		for i := range m.predictionForm.inputs {
			if i == m.predictionForm.focusIndex {
				cmds[i] = m.predictionForm.inputs[i].Focus()
				m.predictionForm.inputs[i].PromptStyle = focusedStyle
				m.predictionForm.inputs[i].TextStyle = focusedStyle
				continue
			}
			m.predictionForm.inputs[i].Blur()
			m.predictionForm.inputs[i].PromptStyle = lipgloss.NewStyle()
			m.predictionForm.inputs[i].TextStyle = lipgloss.NewStyle()
		}

		return m, tea.Batch(cmds...)
	}

	cmd := m.updatePredictionInputs(msg)
	return m, cmd
}

func (m *Model) updatePredictionInputs(msg tea.Msg) tea.Cmd {
	cmds := make([]tea.Cmd, len(m.predictionForm.inputs))

	for i := range m.predictionForm.inputs {
		if m.predictionForm.inputs[i].Focused() {
			m.predictionForm.inputs[i], cmds[i] = m.predictionForm.inputs[i].Update(msg)
		}
	}

	return tea.Batch(cmds...)
}

func (m *Model) viewPredictionForm() string {
	var b strings.Builder

	b.WriteString("Load Model & Predict\n\n")

	b.WriteString("Available Models:\n")
	if len(m.predictionForm.models) == 0 {
		b.WriteString("  (No models found in saved_models/)\n")
	} else {
		for i, model := range m.predictionForm.models {
			b.WriteString(fmt.Sprintf("  %d: %s\n", i+1, filepath.Base(model)))
		}
	}
	b.WriteString("\n")

	fmt.Fprintf(&b, "Select Model (number): %s\n", m.predictionForm.inputs[0].View())
	fmt.Fprintf(&b, "Input Data (comma-separated): %s\n", m.predictionForm.inputs[1].View())
	b.WriteString("\n")

	button := "[ Predict ]"
	if m.predictionForm.focusIndex == len(m.predictionForm.inputs) {
		b.WriteString(focusedStyle.Render(button))
	} else {
		b.WriteString(button)
	}

	b.WriteString(helpStyle.Render("\n\n  ↑/↓, tab/shift+tab: navigate | enter: select | q: back\n"))

	return b.String()
}

func (m *Model) runPrediction() tea.Cmd {
	return func() tea.Msg {
		modelIndex, err := strconv.Atoi(m.predictionForm.inputs[0].Value())
		if err != nil || modelIndex < 1 || modelIndex > len(m.predictionForm.models) {
			return errorMsg{fmt.Errorf("invalid model selection")}
		}
		modelPath := m.predictionForm.models[modelIndex-1]

		modelData, err := data.LoadModel(modelPath)
		if err != nil {
			return errorMsg{fmt.Errorf("failed to load model: %w", err)}
		}
		if err := modelData.NN.SetActivationFunctions(); err != nil {
			return errorMsg{fmt.Errorf("failed to set activation functions: %w", err)}
		}

		inputStrs := strings.Split(strings.TrimSpace(m.predictionForm.inputs[1].Value()), ",")
		if len(inputStrs) != modelData.NN.NumInputs {
			return errorMsg{fmt.Errorf("expected %d input values, but got %d", modelData.NN.NumInputs, len(inputStrs))}
		}

		predictionInput := make([]float64, modelData.NN.NumInputs)
		for i, s := range inputStrs {
			val, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
			if err != nil {
				return errorMsg{fmt.Errorf("invalid input value: %v", err)}
			}
			predictionInput[i] = (val - modelData.InputMins[i]) / (modelData.InputMaxs[i] - modelData.InputMins[i])
		}

		_, predictionOutput := modelData.NN.FeedForward(predictionInput)

		if modelData.ClassMap != nil {
			// Classification
			max := -1.0
			maxIndex := -1
			for i, val := range predictionOutput {
				if val > max {
					max = val
					maxIndex = i
				}
			}
			for class, index := range modelData.ClassMap {
				if index == maxIndex {
					return predictionResultClassificationMsg{result: class}
				}
			}
			return errorMsg{fmt.Errorf("could not determine class from prediction")}
		} else {
			// Regression
			finalPrediction := predictionOutput[0]*(modelData.TargetMaxs[0]-modelData.TargetMins[0]) + modelData.TargetMins[0]
			return predictionResultMsg{result: finalPrediction}
		}
	}
}

func (m *Model) viewPredictionResult() string {
	if m.predictionClass != "" {
		return fmt.Sprintf("Prediction Result: %s\n\n(Press enter to return to main menu)", m.predictionClass)
	}
	return fmt.Sprintf("Prediction Result: %f\n\n(Press enter to return to main menu)", m.predictionValue)
}

func (m *Model) updateSaveModelForm(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg.String() {
	case "enter":
		modelName := m.saveModelInput.Value()
		if modelName != "" {
			modelPath := filepath.Join("saved_models", modelName+".json")
			if err := m.modelData.SaveModel(modelPath); err != nil {
				return m, func() tea.Msg { return errorMsg{err} }
			}
		}
		m.state = mainMenu
		return m, nil

	case "ctrl+c", "q":
		m.state = mainMenu
		return m, nil
	}

	m.saveModelInput, cmd = m.saveModelInput.Update(msg)
	return m, cmd
}

func (m *Model) viewSaveModelForm() string {
	return fmt.Sprintf(
		"Training complete!\n\nEnter a name to save this model (or press Enter to skip):\n\n%s\n\n%s",
		m.saveModelInput.View(),
		helpStyle.Render("enter: save | q: skip"),
	)
}

func (m *Model) viewError() string {
	return fmt.Sprintf(
		"An error occurred:\n\n%s\n\n%s",
		errorStyle.Render(m.lastError.Error()),
		helpStyle.Render("Press enter or q to return to the main menu."),
	)
}

func Start() {
	m := New()
	p := tea.NewProgram(m, tea.WithAltScreen())
	m.program = p

	if _, err := p.Run(); err != nil {
		fmt.Printf("Alas, there's been an error: %v", err)
		os.Exit(1)
	}
}
