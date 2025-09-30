package tempfile

import (
	"os"
)

// CreateTempFileWithContent creates a temporary file with the given content and pattern.
// It uses os.CreateTemp, which is the modern replacement for the deprecated ioutil.TempFile.
// The function handles creating the file, writing the content, and closing it.
// It returns the path to the created file, which should be removed by the caller
// when it's no longer needed.
func CreateTempFileWithContent(pattern, content string) (string, error) {
	// Create a temporary file. The first argument being an empty string
	// means it will be created in the default directory for temporary files.
	tmpfile, err := os.CreateTemp("", pattern)
	if err != nil {
		return "", err
	}
	// Defer closing the file to ensure it's closed even if a write error occurs.
	defer tmpfile.Close()

	// Write the provided content to the file.
	if _, err := tmpfile.WriteString(content); err != nil {
		// If writing fails, we still return the file name, but also the error.
		// The caller might still want to attempt to clean up the empty file.
		return tmpfile.Name(), err
	}

	// Return the name of the file and nil for the error, indicating success.
	return tmpfile.Name(), nil
}
