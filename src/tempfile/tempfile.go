// Package tempfile creates the output models
package tempfile

import (
	"os"
)

func CreateTempFileWithContent(pattern, content string) (string, error) {
	// Create a temporary file. The first argument being an empty string
	// means it will be created in the default directory for temporary files.
	tmpfile, err := os.CreateTemp("", pattern)
	if err != nil {
		return "", err
	}

	defer func() {
		_ = tmpfile.Close()
	}()

	// Write the provided content to the file.
	if _, err := tmpfile.WriteString(content); err != nil {
		// If writing fails, we still return the file name, but also the error.
		return tmpfile.Name(), err
	}

	// Return the name of the file and nil for the error, indicating success.
	return tmpfile.Name(), nil
}
