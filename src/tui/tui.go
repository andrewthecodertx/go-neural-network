// Package tui creates the console interaction
package tui

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"go-neuralnetwork/src/data"
	"go-neuralnetwork/src/engine"
	"go-neuralnetwork/src/neuralnetwork"
	"go-neuralnetwork/src/visualization"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// ─── Messages ───────────────────────────────────────────────────────────────

type (
	csvFilesLoadedMsg  struct{ files []string }
	modelsLoadedMsg    struct{ models []string }
	trainingStartedMsg struct{}
	epochCompletedMsg  struct {
		epochNum int
		loss     float64
	}
	trainingFinishedMsg struct {
		result engine.TrainResult
	}
	evaluationFinishedMsg struct{ result data.EvaluationResult }
	predictionFinishedMsg struct{ result engine.PredictionResult }
	errorMsg              struct{ err error }
)

func waitForEpoch(progressChan <-chan float64, epochNum int, doneChan <-chan engine.TrainResult) tea.Cmd {
	return func() tea.Msg {
		select {
		case loss, ok := <-progressChan:
			if !ok {
				return trainingFinishedMsg{result: <-doneChan}
			}
			return epochCompletedMsg{epochNum: epochNum, loss: loss}
		case res := <-doneChan:
			return trainingFinishedMsg{result: res}
		}
	}
}

// ─── Styles ─────────────────────────────────────────────────────────────────

var (
	brandPurple  = lipgloss.Color("#7D56F4")
	brandBlue    = lipgloss.Color("#5B9FD6")
	dimText      = lipgloss.Color("#666666")
	successGreen = lipgloss.Color("#4EC9B0")
	warnOrange   = lipgloss.Color("#FF8700")

	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FAFAFA")).
			Background(brandPurple).
			Padding(0, 2)

	panelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(brandPurple).
			Padding(1, 2)

	helpStyle = lipgloss.NewStyle().
			Foreground(dimText).
			Padding(1, 0, 0, 2)

	selectedStyle = lipgloss.NewStyle().
			Foreground(brandPurple).
			Bold(true)

	unselectedStyle = lipgloss.NewStyle().
			Foreground(dimText)

	focusedLabel = lipgloss.NewStyle().
			Foreground(brandBlue).
			Bold(true)

	unfocusedLabel = lipgloss.NewStyle().
			Foreground(dimText)

	buttonStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FAFAFA")).
			Background(brandPurple).
			Padding(0, 3).
			Bold(true)

	buttonBlurredStyle = lipgloss.NewStyle().
				Foreground(dimText).
				Padding(0, 3)

	errorStyle = lipgloss.NewStyle().
			Foreground(warnOrange).
			Bold(true)

	successStyle = lipgloss.NewStyle().
			Foreground(successGreen).
			Bold(true)

	cycleValueStyle = lipgloss.NewStyle().
			Foreground(brandBlue).
			Bold(true)

	cycleHintStyle = lipgloss.NewStyle().
			Foreground(dimText)
)

// ─── Field types ────────────────────────────────────────────────────────────

// fieldKind distinguishes interactive field types in the training form.
type fieldKind int

const (
	fieldList   fieldKind = iota // arrow-up/down to cycle through options
	fieldInput                   // freeform text input
	fieldCycle                   // left/right to cycle through values
	fieldButton                  // action button (enter to activate)
)

// listField is a scrollable selector over a list of string options.
type listField struct {
	label   string
	options []string
	cursor  int
}

func (f *listField) up() {
	if f.cursor > 0 {
		f.cursor--
	}
}
func (f *listField) down() {
	if f.cursor < len(f.options)-1 {
		f.cursor++
	}
}
func (f *listField) value() string {
	if len(f.options) == 0 {
		return ""
	}
	return f.options[f.cursor]
}

// cycleField is a left/right toggle over a list of string options.
type cycleField struct {
	label   string
	options []string
	cursor  int
}

func (f *cycleField) left() {
	if f.cursor > 0 {
		f.cursor--
	}
}
func (f *cycleField) right() {
	if f.cursor < len(f.options)-1 {
		f.cursor++
	}
}
func (f *cycleField) value() string {
	return f.options[f.cursor]
}

// ─── Application state ──────────────────────────────────────────────────────

type sessionState int

const (
	stateMainMenu sessionState = iota
	stateTrainForm
	stateTrainProgress
	stateTrainViz
	stateEvalResult
	stateSaveModel
	statePredictForm
	statePredictResult
	stateError
)

// Model represents the entire application state.
type Model struct {
	state    sessionState
	width    int
	height   int
	quitting bool

	// Main menu
	menuCursor int

	// Training form fields — order matters, it defines tab order.
	trainFields []fieldKind
	datasetList listField       // 0: dataset selector
	layerInput  textinput.Model // 1: hidden layers e.g. "20,20"
	hiddenCycle cycleField      // 2: hidden activations (cycles through presets)
	outputCycle cycleField      // 3: output activation
	epochsInput textinput.Model // 4: epochs
	lrInput     textinput.Model // 5: learning rate
	egInput     textinput.Model // 6: error goal
	vizCycle    cycleField      // 7: visualization on/off
	focusIndex  int             // which field has focus (0..len(trainFields))

	// Training progress
	progressChan <-chan float64
	doneChan     <-chan engine.TrainResult
	currentEpoch int
	totalEpochs  int
	lastLoss     float64
	progBar      progress.Model

	// Evaluation
	modelData        *data.ModelData
	evaluationResult data.EvaluationResult

	// Prediction form
	predictModelList listField
	predictInput     textinput.Model
	predictFocus     int // 0=model list, 1=input, 2=predict button

	// Prediction result
	predictionValue float64
	predictionClass string

	// Save
	saveInput textinput.Model

	// Error
	lastError error
}

// ─── Constructor ─────────────────────────────────────────────────────────────

func New() *Model {
	// Text inputs with sensible defaults
	li := textinput.New()
	li.Placeholder = "20,20"
	li.Prompt = ""
	li.CharLimit = 64

	ei := textinput.New()
	ei.Placeholder = "1000"
	ei.Prompt = ""
	ei.CharLimit = 8

	lri := textinput.New()
	lri.Placeholder = "0.001"
	lri.Prompt = ""
	lri.CharLimit = 16

	egi := textinput.New()
	egi.Placeholder = "0.001"
	egi.Prompt = ""
	egi.CharLimit = 16

	pi := textinput.New()
	pi.Placeholder = "5.1,3.5,1.4,0.2"
	pi.Prompt = ""
	pi.CharLimit = 256

	si := textinput.New()
	si.Placeholder = "my-model"
	si.Prompt = ""
	si.CharLimit = 64
	si.Width = 40

	avail := neuralnetwork.GetAvailableActivations()

	// Activation presets: each preset is a comma-separated list matching the
	// number of hidden layers the user will specify. For simplicity we offer
	// common patterns; the user can always type a custom layers spec.
	hiddenPresets := make([]string, len(avail))
	for i, a := range avail {
		hiddenPresets[i] = a
	}
	outputPresets := avail
	boolCycle := []string{"no", "yes"}

	pb := progress.New(progress.WithGradient(string(brandPurple), string(successGreen)))

	trainFields := []fieldKind{
		fieldList,   // dataset
		fieldInput,  // hidden layers
		fieldCycle,  // hidden activation
		fieldCycle,  // output activation
		fieldInput,  // epochs
		fieldInput,  // learning rate
		fieldInput,  // error goal
		fieldCycle,  // visualization
		fieldButton, // start training
	}

	return &Model{
		state:        stateMainMenu,
		menuCursor:   0,
		trainFields:  trainFields,
		datasetList:  listField{label: "Dataset"},
		layerInput:   li,
		hiddenCycle:  cycleField{label: "Hidden Activation", options: hiddenPresets, cursor: 0},
		outputCycle:  cycleField{label: "Output Activation", options: outputPresets, cursor: len(outputPresets) - 1}, // default "linear"
		epochsInput:  ei,
		lrInput:      lri,
		egInput:      egi,
		vizCycle:     cycleField{label: "Visualization", options: boolCycle, cursor: 0},
		focusIndex:   0,
		progBar:      pb,
		predictInput: pi,
		saveInput:    si,
	}
}

// ─── Init ────────────────────────────────────────────────────────────────────

func (m *Model) Init() tea.Cmd {
	return textinput.Blink
}

// ─── Update ──────────────────────────────────────────────────────────────────

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.progBar.Width = msg.Width - 4
		if m.progBar.Width > 60 {
			m.progBar.Width = 60
		}
		return m, nil

	case csvFilesLoadedMsg:
		m.datasetList.options = msg.files
		m.datasetList.cursor = 0
		return m, nil

	case modelsLoadedMsg:
		m.predictModelList.options = msg.models
		m.predictModelList.cursor = 0
		return m, nil

	case errorMsg:
		m.lastError = msg.err
		m.state = stateError
		return m, nil

	case trainingStartedMsg:
		if m.vizCycle.value() == "yes" {
			m.state = stateTrainViz
		} else {
			m.state = stateTrainProgress
		}
		return m, waitForEpoch(m.progressChan, 1, m.doneChan)

	case epochCompletedMsg:
		m.currentEpoch = msg.epochNum
		m.lastLoss = msg.loss
		return m, waitForEpoch(m.progressChan, msg.epochNum+1, m.doneChan)

	case trainingFinishedMsg:
		m.modelData = msg.result.Model
		m.state = stateEvalResult
		result := msg.result
		return m, func() tea.Msg {
			return evaluationFinishedMsg{result: engine.Evaluate(result)}
		}

	case evaluationFinishedMsg:
		m.evaluationResult = msg.result
		m.state = stateSaveModel
		return m, nil

	case predictionFinishedMsg:
		m.state = statePredictResult
		if msg.result.IsClassification {
			m.predictionClass = msg.result.Class
		} else {
			m.predictionValue = msg.result.Value
		}
		return m, nil

	case progress.FrameMsg:
		progressModel, cmd := m.progBar.Update(msg)
		m.progBar = progressModel.(progress.Model)
		return m, cmd
	}

	// State-specific key handling
	switch m.state {
	case stateMainMenu:
		return m.updateMainMenu(msg)
	case stateTrainForm:
		return m.updateTrainForm(msg)
	case stateTrainProgress, stateTrainViz:
		return m.updateTrainProgress(msg)
	case stateEvalResult:
		return m.updateEvalResult(msg)
	case stateSaveModel:
		return m.updateSaveModel(msg)
	case statePredictForm:
		return m.updatePredictForm(msg)
	case statePredictResult:
		return m.updatePredictResult(msg)
	case stateError:
		return m.updateError(msg)
	}
	return m, nil
}

// ─── Main Menu ──────────────────────────────────────────────────────────────

func (m *Model) updateMainMenu(msg tea.Msg) (tea.Model, tea.Cmd) {
	key, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	switch key.String() {
	case "ctrl+c", "q":
		m.quitting = true
		return m, tea.Quit
	case "up", "k":
		if m.menuCursor > 0 {
			m.menuCursor--
		}
	case "down", "j":
		if m.menuCursor < 2 {
			m.menuCursor++
		}
	case "enter":
		switch m.menuCursor {
		case 0:
			m.state = stateTrainForm
			m.focusIndex = 0
			m.layerInput.Focus()
			return m, findCsvFiles
		case 1:
			m.state = statePredictForm
			m.predictFocus = 0
			return m, findModels
		case 2:
			m.quitting = true
			return m, tea.Quit
		}
	}
	return m, nil
}

// ─── Training Form ──────────────────────────────────────────────────────────

func (m *Model) updateTrainForm(msg tea.Msg) (tea.Model, tea.Cmd) {
	key, ok := msg.(tea.KeyMsg)
	if !ok {
		// Pass through to focused text input
		return m.updateTrainTextInput(msg)
	}

	switch key.String() {
	case "ctrl+c":
		m.quitting = true
		return m, tea.Quit
	case "esc", "q":
		m.state = stateMainMenu
		return m, nil
	case "tab", "down":
		m.focusIndex = (m.focusIndex + 1) % len(m.trainFields)
		m.syncTrainFocus()
		return m, nil
	case "shift+tab", "up":
		m.focusIndex = (m.focusIndex - 1 + len(m.trainFields)) % len(m.trainFields)
		m.syncTrainFocus()
		return m, nil
	case "left":
		if m.focusIndex < len(m.trainFields) {
			switch m.trainFields[m.focusIndex] {
			case fieldList:
				m.datasetList.up()
			case fieldCycle:
				m.activeCycle().left()
			}
		}
		return m, nil
	case "right":
		if m.focusIndex < len(m.trainFields) {
			switch m.trainFields[m.focusIndex] {
			case fieldList:
				m.datasetList.down()
			case fieldCycle:
				m.activeCycle().right()
			}
		}
		return m, nil
	case "enter":
		if m.focusIndex == len(m.trainFields)-1 {
			// Start Training button
			return m, m.runTraining()
		}
		// For list/cycle fields, enter also moves forward like tab
		if m.trainFields[m.focusIndex] == fieldList || m.trainFields[m.focusIndex] == fieldCycle {
			m.focusIndex = (m.focusIndex + 1) % len(m.trainFields)
			m.syncTrainFocus()
			return m, nil
		}
		// Text input fields: enter moves to next field
		m.focusIndex = (m.focusIndex + 1) % len(m.trainFields)
		m.syncTrainFocus()
		return m, nil
	}

	// If a text input is focused, pass the key through
	return m.updateTrainTextInput(msg)
}

func (m *Model) activeCycle() *cycleField {
	switch m.focusIndex {
	case 2:
		return &m.hiddenCycle
	case 3:
		return &m.outputCycle
	case 7:
		return &m.vizCycle
	}
	return nil
}

func (m *Model) syncTrainFocus() {
	// Blur all text inputs
	m.layerInput.Blur()
	m.epochsInput.Blur()
	m.lrInput.Blur()
	m.egInput.Blur()
	// Focus the active text input if any
	for i, kind := range m.trainFields {
		if i != m.focusIndex {
			continue
		}
		switch i {
		case 1:
			m.layerInput.Focus()
		case 4:
			m.epochsInput.Focus()
		case 5:
			m.lrInput.Focus()
		case 6:
			m.egInput.Focus()
		}
		_ = kind
		break
	}
}

func (m *Model) updateTrainTextInput(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	switch m.focusIndex {
	case 1:
		m.layerInput, cmd = m.layerInput.Update(msg)
	case 4:
		m.epochsInput, cmd = m.epochsInput.Update(msg)
	case 5:
		m.lrInput, cmd = m.lrInput.Update(msg)
	case 6:
		m.egInput, cmd = m.egInput.Update(msg)
	}
	return m, cmd
}

func (m *Model) runTraining() tea.Cmd {
	return func() tea.Msg {
		csvPath := m.datasetList.value()
		if csvPath == "" {
			return errorMsg{fmt.Errorf("no dataset selected")}
		}

		raw := engine.RawTrainConfig{
			CSVPath:           csvPath,
			HiddenLayers:      m.layerInput.Value(),
			HiddenActivations: m.hiddenCycle.value(),
			OutputActivation:  m.outputCycle.value(),
			Epochs:            m.epochsInput.Value(),
			LearningRate:      m.lrInput.Value(),
			ErrorGoal:         m.egInput.Value(),
			EnableViz:         m.vizCycle.value() == "yes",
		}

		cfg, err := engine.ParseTrainConfig(raw)
		if err != nil {
			return errorMsg{err}
		}

		handles, err := engine.StartTraining(cfg)
		if err != nil {
			return errorMsg{err}
		}

		m.progressChan = handles.Progress
		m.doneChan = handles.Done
		m.totalEpochs = cfg.Epochs

		if handles.Viz != nil {
			startVisualizer(handles.Network, handles.Viz)
		}

		return trainingStartedMsg{}
	}
}

// ─── Training Progress ───────────────────────────────────────────────────────

func (m *Model) updateTrainProgress(msg tea.Msg) (tea.Model, tea.Cmd) {
	key, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	if key.String() == "q" || key.String() == "esc" {
		m.state = stateMainMenu
		return m, nil
	}
	return m, nil
}

// ─── Evaluation Result ───────────────────────────────────────────────────────

func (m *Model) updateEvalResult(msg tea.Msg) (tea.Model, tea.Cmd) {
	key, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	if key.String() == "enter" || key.String() == "q" || key.String() == "esc" {
		m.state = stateSaveModel
		m.saveInput.Focus()
		return m, nil
	}
	return m, nil
}

// ─── Save Model ─────────────────────────────────────────────────────────────

func (m *Model) updateSaveModel(msg tea.Msg) (tea.Model, tea.Cmd) {
	key, ok := msg.(tea.KeyMsg)
	if ok {
		switch key.String() {
		case "ctrl+c":
			m.quitting = true
			return m, tea.Quit
		case "esc", "q":
			m.state = stateMainMenu
			return m, nil
		case "enter":
			name := m.saveInput.Value()
			if name != "" {
				if _, err := engine.SaveModel(m.modelData, name); err != nil {
					return m, func() tea.Msg { return errorMsg{err} }
				}
			}
			m.state = stateMainMenu
			return m, nil
		}
	}
	var cmd tea.Cmd
	m.saveInput, cmd = m.saveInput.Update(msg)
	return m, cmd
}

// ─── Predict Form ───────────────────────────────────────────────────────────

func (m *Model) updatePredictForm(msg tea.Msg) (tea.Model, tea.Cmd) {
	key, ok := msg.(tea.KeyMsg)
	if !ok {
		return m.updatePredictInput(msg)
	}

	switch key.String() {
	case "ctrl+c":
		m.quitting = true
		return m, tea.Quit
	case "esc", "q":
		m.state = stateMainMenu
		return m, nil
	case "tab", "down":
		m.predictFocus = (m.predictFocus + 1) % 3
		m.syncPredictFocus()
		return m, nil
	case "shift+tab", "up":
		m.predictFocus = (m.predictFocus - 1 + 3) % 3
		m.syncPredictFocus()
		return m, nil
	case "left":
		if m.predictFocus == 0 {
			m.predictModelList.up()
		}
		return m, nil
	case "right":
		if m.predictFocus == 0 {
			m.predictModelList.down()
		}
		return m, nil
	case "enter":
		if m.predictFocus == 2 {
			return m, m.runPrediction()
		}
		m.predictFocus = (m.predictFocus + 1) % 3
		m.syncPredictFocus()
		return m, nil
	}

	return m.updatePredictInput(msg)
}

func (m *Model) syncPredictFocus() {
	if m.predictFocus == 1 {
		m.predictInput.Focus()
	} else {
		m.predictInput.Blur()
	}
}

func (m *Model) updatePredictInput(msg tea.Msg) (tea.Model, tea.Cmd) {
	if m.predictFocus != 1 {
		return m, nil
	}
	var cmd tea.Cmd
	m.predictInput, cmd = m.predictInput.Update(msg)
	return m, cmd
}

func (m *Model) runPrediction() tea.Cmd {
	return func() tea.Msg {
		if len(m.predictModelList.options) == 0 {
			return errorMsg{fmt.Errorf("no model selected")}
		}
		modelData, err := engine.LoadModelForPrediction(m.predictModelList.value())
		if err != nil {
			return errorMsg{err}
		}

		rawInput, err := engine.ParseInputVector(m.predictInput.Value())
		if err != nil {
			return errorMsg{err}
		}

		result, err := engine.Predict(modelData, rawInput)
		if err != nil {
			return errorMsg{err}
		}
		return predictionFinishedMsg{result: result}
	}
}

// ─── Predict Result ─────────────────────────────────────────────────────────

func (m *Model) updatePredictResult(msg tea.Msg) (tea.Model, tea.Cmd) {
	key, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	if key.String() == "enter" || key.String() == "q" || key.String() == "esc" {
		m.state = stateMainMenu
		return m, nil
	}
	return m, nil
}

// ─── Error ───────────────────────────────────────────────────────────────────

func (m *Model) updateError(msg tea.Msg) (tea.Model, tea.Cmd) {
	key, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}
	if key.String() == "enter" || key.String() == "q" || key.String() == "esc" {
		m.state = stateMainMenu
		return m, nil
	}
	return m, nil
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

func findCsvFiles() tea.Msg {
	files, err := engine.FindDatasets()
	if err != nil {
		return errorMsg{err}
	}
	return csvFilesLoadedMsg{files}
}

func findModels() tea.Msg {
	files, err := engine.FindModels()
	if err != nil {
		return errorMsg{err}
	}
	return modelsLoadedMsg{models: files}
}

func startVisualizer(nn *neuralnetwork.NeuralNetwork, vizChan <-chan [][]float64) {
	go func() {
		runtime.LockOSThread()
		viz, err := visualization.NewNetworkVisualizer(900, 600)
		if err != nil {
			for range vizChan {
			}
			return
		}
		defer viz.Close()
		for activations := range vizChan {
			viz.HandleEvents()
			if !viz.IsRunning() {
				for range vizChan {
				}
				return
			}
			viz.RenderNetwork(nn, activations)
		}
	}()
}

// ─── View ────────────────────────────────────────────────────────────────────

func (m *Model) View() string {
	if m.quitting {
		return "\n  Goodbye!\n\n"
	}

	switch m.state {
	case stateMainMenu:
		return m.viewMainMenu()
	case stateTrainForm:
		return m.viewTrainForm()
	case stateTrainProgress:
		return m.viewTrainProgress()
	case stateTrainViz:
		return m.viewTrainViz()
	case stateEvalResult:
		return m.viewEvalResult()
	case stateSaveModel:
		return m.viewSaveModel()
	case statePredictForm:
		return m.viewPredictForm()
	case statePredictResult:
		return m.viewPredictResult()
	case stateError:
		return m.viewError()
	default:
		return "Unknown state."
	}
}

func (m *Model) viewMainMenu() string {
	choices := []string{"Train New Model", "Load Model & Predict", "Quit"}

	var b strings.Builder
	b.WriteString(titleStyle.Render(" Go Neural Network "))
	b.WriteString("\n\n")

	for i, choice := range choices {
		if m.menuCursor == i {
			b.WriteString(selectedStyle.Render(fmt.Sprintf("  ▸ %s", choice)))
		} else {
			b.WriteString(unselectedStyle.Render(fmt.Sprintf("    %s", choice)))
		}
		b.WriteString("\n")
	}

	b.WriteString("\n")
	b.WriteString(helpStyle.Render("↑/↓  navigate · enter  select · q  quit"))
	return b.String()
}

func (m *Model) viewTrainForm() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render(" Train New Model "))
	b.WriteString("\n\n")

	// ── Dataset selector ──
	b.WriteString(m.renderListField("Dataset", m.datasetList, 0))
	b.WriteString("\n")

	// ── Architecture ──
	b.WriteString(lipgloss.NewStyle().Bold(true).Foreground(brandPurple).Render("─ Architecture ─"))
	b.WriteString("\n\n")

	b.WriteString(m.renderInputField("Hidden Layers", m.layerInput, 1, "e.g. 20,20"))
	b.WriteString("\n")

	// Number of hidden layers hint
	layersStr := strings.TrimSpace(m.layerInput.Value())
	if layersStr == "" {
		layersStr = "20,20"
	}
	layerCount := len(strings.Split(layersStr, ","))
	b.WriteString(cycleHintStyle.Render(fmt.Sprintf("  (%d layer%s — activation will be repeated)", layerCount, pluralS(layerCount))))
	b.WriteString("\n")

	b.WriteString(m.renderCycleField("Hidden Activation", m.hiddenCycle, 2))
	b.WriteString("\n")

	b.WriteString(m.renderCycleField("Output Activation", m.outputCycle, 3))
	b.WriteString("\n")

	// ── Training parameters ──
	b.WriteString(lipgloss.NewStyle().Bold(true).Foreground(brandPurple).Render("─ Parameters ──"))
	b.WriteString("\n\n")

	b.WriteString(m.renderInputField("Epochs", m.epochsInput, 4, "1000"))
	b.WriteString("\n")

	b.WriteString(m.renderInputField("Learning Rate", m.lrInput, 5, "0.001"))
	b.WriteString("\n")

	b.WriteString(m.renderInputField("Error Goal", m.egInput, 6, "0.001"))
	b.WriteString("\n")

	b.WriteString(m.renderCycleField("Visualization", m.vizCycle, 7))
	b.WriteString("\n\n")

	// ── Start button ──
	if m.focusIndex == len(m.trainFields)-1 {
		b.WriteString(buttonStyle.Render("▶ Start Training"))
	} else {
		b.WriteString(buttonBlurredStyle.Render("  Start Training"))
	}

	b.WriteString("\n")
	b.WriteString(helpStyle.Render("tab/↑↓  navigate · ←/→  change · enter  select/start · esc  back"))
	return b.String()
}

func (m *Model) viewTrainProgress() string {
	pct := 0.0
	if m.totalEpochs > 0 {
		pct = float64(m.currentEpoch) / float64(m.totalEpochs)
	}
	if pct > 1 {
		pct = 1
	}

	var b strings.Builder
	b.WriteString(titleStyle.Render(" Training "))
	b.WriteString("\n\n")

	b.WriteString(fmt.Sprintf("  Epoch  %d / %d\n", m.currentEpoch, m.totalEpochs))
	b.WriteString(fmt.Sprintf("  Loss   %.6f\n\n", m.lastLoss))

	b.WriteString(m.progBar.ViewAs(pct))
	b.WriteString("\n\n")

	b.WriteString(helpStyle.Render("q/esc  cancel · window auto-closes when done"))
	return b.String()
}

func (m *Model) viewTrainViz() string {
	pct := 0.0
	if m.totalEpochs > 0 {
		pct = float64(m.currentEpoch) / float64(m.totalEpochs)
	}
	if pct > 1 {
		pct = 1
	}

	var b strings.Builder
	b.WriteString(titleStyle.Render(" Training + Visualization "))
	b.WriteString("\n\n")

	b.WriteString(fmt.Sprintf("  Epoch  %d / %d\n", m.currentEpoch, m.totalEpochs))
	b.WriteString(fmt.Sprintf("  Loss   %.6f\n\n", m.lastLoss))

	b.WriteString(m.progBar.ViewAs(pct))
	b.WriteString("\n\n")

	b.WriteString(panelStyle.Render(
		"SDL2 visualization window is active.\n\n" +
			"  • Colored nodes show activations\n" +
			"  • Red/Blue lines show positive/negative weights\n" +
			"  • Press ESC in the SDL window to close it"))
	b.WriteString("\n\n")

	b.WriteString(helpStyle.Render("q/esc  cancel · window auto-closes when done"))
	return b.String()
}

func (m *Model) viewEvalResult() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render(" Evaluation Complete "))
	b.WriteString("\n\n")

	r := m.evaluationResult
	if r.IsClassification {
		acc := fmt.Sprintf("%.2f%%", r.Accuracy*100)
		b.WriteString(fmt.Sprintf("  Accuracy:  %s\n", successStyle.Render(acc)))
	} else {
		b.WriteString(fmt.Sprintf("  RMSE:  %.6f\n", r.RMSE))
		b.WriteString(fmt.Sprintf("  R²:    %.4f\n", r.RSquared))
	}

	b.WriteString("\n")
	b.WriteString(helpStyle.Render("enter  continue · save model"))
	return b.String()
}

func (m *Model) viewSaveModel() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render(" Save Model "))
	b.WriteString("\n\n")

	b.WriteString("  Enter a name to save (or leave blank to skip):\n\n  ")
	b.WriteString(m.saveInput.View())
	b.WriteString("\n\n")

	b.WriteString(helpStyle.Render("enter  save · esc/q  skip"))
	return b.String()
}

func (m *Model) viewPredictForm() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render(" Load Model & Predict "))
	b.WriteString("\n\n")

	// Model selector
	b.WriteString(m.renderPredictModelList())
	b.WriteString("\n")

	b.WriteString(m.renderPredictInputField())
	b.WriteString("\n")

	// Predict button
	if m.predictFocus == 2 {
		b.WriteString(buttonStyle.Render("▶ Predict"))
	} else {
		b.WriteString(buttonBlurredStyle.Render("  Predict"))
	}
	b.WriteString("\n\n")

	b.WriteString(helpStyle.Render("tab/↑↓  navigate · ←/→  select model · enter  predict · esc  back"))
	return b.String()
}

func (m *Model) viewPredictResult() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render(" Prediction Result "))
	b.WriteString("\n\n")

	if m.predictionClass != "" {
		b.WriteString(successStyle.Render(fmt.Sprintf("  Class:  %s", m.predictionClass)))
	} else {
		b.WriteString(successStyle.Render(fmt.Sprintf("  Value:  %.6f", m.predictionValue)))
	}

	b.WriteString("\n\n")
	b.WriteString(helpStyle.Render("enter/esc  return to menu"))
	return b.String()
}

func (m *Model) viewError() string {
	var b strings.Builder
	b.WriteString(titleStyle.Render(" Error "))
	b.WriteString("\n\n")

	b.WriteString(errorStyle.Render(fmt.Sprintf("  %s", m.lastError.Error())))
	b.WriteString("\n\n")
	b.WriteString(helpStyle.Render("enter/esc  return to menu"))
	return b.String()
}

// ─── View helpers ────────────────────────────────────────────────────────────

func (m *Model) renderListField(label string, lf listField, fieldIdx int) string {
	isFocused := m.focusIndex == fieldIdx
	labelStyle := unfocusedLabel
	if isFocused {
		labelStyle = focusedLabel
	}

	var b strings.Builder
	b.WriteString(labelStyle.Render(label + ":"))
	b.WriteString("\n")

	for i, opt := range lf.options {
		name := filepath.Base(opt)
		if isFocused && i == lf.cursor {
			b.WriteString(selectedStyle.Render(fmt.Sprintf("    ▸ %s", name)))
		} else if i == lf.cursor {
			b.WriteString(fmt.Sprintf("    ▸ %s", name))
		} else {
			b.WriteString(unselectedStyle.Render(fmt.Sprintf("      %s", name)))
		}
		b.WriteString("\n")
	}
	if len(lf.options) == 0 {
		b.WriteString(unselectedStyle.Render("    (no datasets found)"))
		b.WriteString("\n")
	}
	if isFocused {
		b.WriteString(cycleHintStyle.Render("  ↑↓ to select"))
	}
	return b.String()
}

func (m *Model) renderCycleField(label string, cf cycleField, fieldIdx int) string {
	isFocused := m.focusIndex == fieldIdx
	labelStyle := unfocusedLabel
	if isFocused {
		labelStyle = focusedLabel
	}

	val := cf.value()
	if isFocused {
		return labelStyle.Render(label+":") + " " + cycleValueStyle.Render(fmt.Sprintf("◀ %s ▶", val))
	}
	return labelStyle.Render(label+":") + " " + val
}

func (m *Model) renderInputField(label string, ti textinput.Model, fieldIdx int, placeholder string) string {
	isFocused := m.focusIndex == fieldIdx
	labelStyle := unfocusedLabel
	if isFocused {
		labelStyle = focusedLabel
	}

	// Show placeholder as hint when empty and unfocused
	hint := ""
	if ti.Value() == "" && !isFocused {
		hint = cycleHintStyle.Render(fmt.Sprintf(" (%s)", placeholder))
	}

	return labelStyle.Render(label+":") + " " + ti.View() + hint
}

func (m *Model) renderPredictModelList() string {
	isFocused := m.predictFocus == 0
	labelStyle := unfocusedLabel
	if isFocused {
		labelStyle = focusedLabel
	}

	var b strings.Builder
	b.WriteString(labelStyle.Render("Model:"))
	b.WriteString("\n")

	for i, opt := range m.predictModelList.options {
		name := filepath.Base(opt)
		if i == m.predictModelList.cursor {
			if isFocused {
				b.WriteString(selectedStyle.Render(fmt.Sprintf("  ▸ %s", name)))
			} else {
				b.WriteString(fmt.Sprintf("  ▸ %s", name))
			}
		} else {
			b.WriteString(unselectedStyle.Render(fmt.Sprintf("    %s", name)))
		}
		b.WriteString("\n")
	}
	if len(m.predictModelList.options) == 0 {
		b.WriteString(unselectedStyle.Render("  (no saved models found)"))
		b.WriteString("\n")
	}
	return b.String()
}

func (m *Model) renderPredictInputField() string {
	isFocused := m.predictFocus == 1
	labelStyle := unfocusedLabel
	if isFocused {
		labelStyle = focusedLabel
	}
	return labelStyle.Render("Input (comma-separated):") + " " + m.predictInput.View()
}

func pluralS(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// ─── Start ───────────────────────────────────────────────────────────────────

func Start() {
	m := New()
	p := tea.NewProgram(m, tea.WithAltScreen())

	if _, err := p.Run(); err != nil {
		fmt.Printf("Alas, there's been an error: %v", err)
		os.Exit(1)
	}
}
