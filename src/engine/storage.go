package engine

import (
	"path/filepath"

	"go-neuralnetwork/src/data"
)

// Default on-disk locations, relative to the working directory.
const (
	DatasetsDir = "datasets"
	ModelsDir   = "models"
)

// FindDatasets returns the paths of all CSV datasets in DatasetsDir.
func FindDatasets() ([]string, error) {
	return filepath.Glob(filepath.Join(DatasetsDir, "*.csv"))
}

// FindModels returns the paths of all saved models in ModelsDir.
func FindModels() ([]string, error) {
	return filepath.Glob(filepath.Join(ModelsDir, "*.json"))
}

// SaveModel writes a trained model to ModelsDir under the given name (without
// extension) and returns the path it was written to.
func SaveModel(md *data.ModelData, name string) (string, error) {
	path := filepath.Join(ModelsDir, name+".json")
	if err := md.SaveModel(path); err != nil {
		return "", err
	}
	return path, nil
}
