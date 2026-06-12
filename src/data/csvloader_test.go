package data

import "testing"

// TestSplitDataNoAliasOnAppend guards against the slice-aliasing bug where the
// train slice's capacity overlapped the test region, so appending to the train
// set silently overwrote test rows through the shared backing array.
func TestSplitDataNoAliasOnAppend(t *testing.T) {
	inputs := [][]float64{{0}, {1}, {2}, {3}, {4}}
	targets := [][]float64{{0}, {1}, {2}, {3}, {4}}

	trainInputs, trainTargets, testInputs, testTargets := SplitData(inputs, targets, 0.6)

	if len(trainInputs) != 3 || len(testInputs) != 2 {
		t.Fatalf("unexpected split sizes: train=%d test=%d", len(trainInputs), len(testInputs))
	}

	// Capacity must equal length so an append cannot reach into the test region.
	if cap(trainInputs) != len(trainInputs) {
		t.Errorf("trainInputs cap=%d, want %d (no overlap with test)", cap(trainInputs), len(trainInputs))
	}
	if cap(trainTargets) != len(trainTargets) {
		t.Errorf("trainTargets cap=%d, want %d", cap(trainTargets), len(trainTargets))
	}

	// Snapshot the test set, then mutate the train set via append. With the bug,
	// the first test row would be clobbered.
	testInput0Before := testInputs[0][0]
	testTarget0Before := testTargets[0][0]

	trainInputs = append(trainInputs, []float64{99})
	trainTargets = append(trainTargets, []float64{99})

	if testInputs[0][0] != testInput0Before {
		t.Errorf("appending to trainInputs corrupted testInputs[0]: got %v, want %v",
			testInputs[0][0], testInput0Before)
	}
	if testTargets[0][0] != testTarget0Before {
		t.Errorf("appending to trainTargets corrupted testTargets[0]: got %v, want %v",
			testTargets[0][0], testTarget0Before)
	}
}
