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
	res := make(SliceSampleSet, len(inputs))
	for i, in := range inputs {
		res[i] = VectorSample{Input: in, Output: outputs[i]}
	}
	return res
}

// A SampleSet is an abstract list of abstract training
// samples.
type SampleSet interface {
	// Len returns the length of the sample set.
	Len() int

	// Copy creates a shallow copy of the sample set.
	// The training samples themselves needn't be
	// copied.
	Copy() SampleSet

	// Swap swaps the samples at two indices.
	Swap(i, j int)

	// Get gets the sample at the given index.
	GetSample(idx int) interface{}

	// Subset creates a SampleSet which represents the
	// subset of this sample set from the start index
	// (inclusive) to the end index (exclusive).
	Subset(start, end int) SampleSet
}

func ShuffleSampleSet(s SampleSet) {
	for i := 0; i < s.Len(); i++ {
		j := i + rand.Intn(s.Len()-i)
		s.Swap(i, j)
	}
}

// SliceSampleSet is a SampleSet which is backed by a
// slice of training samples.
type SliceSampleSet []interface{}

func (s SliceSampleSet) Len() int {
	return len(s)
}

func (s SliceSampleSet) Copy() SampleSet {
	res := make(SliceSampleSet, len(s))
	copy(res, s)
	return s
}

func (s SliceSampleSet) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s SliceSampleSet) GetSample(idx int) interface{} {
	return s[idx]
}

func (s SliceSampleSet) Subset(start, end int) SampleSet {
	return s[start:end]
}
