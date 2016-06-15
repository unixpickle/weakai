package neuralnet

import (
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
)

// VectorSample represents a supervised training sample
// for a classifier which takes input vectors and makes
// output vectors.
type VectorSample struct {
	// Input is the input given to the classifier.
	Input linalg.Vector

	// Output is the desired output from the classifier.
	Output linalg.Vector
}

// VectorSampleSet creates a SampleSet of VectorSamples
// given a slice of inputs and a slice of corresponding
// outputs for those inputs.
func VectorSampleSet(inputs []linalg.Vector, outputs []linalg.Vector) SampleSet {
	if len(inputs) != len(outputs) {
		panic("input and output counts do not match")
	}
	res := make(SampleSet, len(inputs))
	for i, in := range inputs {
		res[i] = VectorSample{Input: in, Output: outputs[i]}
	}
	return res
}

// SampleSet facilitates the manipulation of abstract
// training samples.
type SampleSet []interface{}

// Copy performs a shallow copy of the SampleSet.
func (s SampleSet) Copy() SampleSet {
	res := make(SampleSet, len(s))
	copy(res, s)
	return s
}

// Shuffle rearranges the samples in a random order.
func (s SampleSet) Shuffle() {
	for i := range s {
		j := i + rand.Intn(len(s)-i)
		s[i], s[j] = s[j], s[i]
	}
}
