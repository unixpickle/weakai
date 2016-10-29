package neuralnet

import (
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
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

// Hash generates a randomly-distributed hash based on
// the vector data.
func (v *VectorSample) Hash() []byte {
	return sgd.HashVectors(v.Input, v.Output)
}

// VectorSampleSet creates an sgd.SampleSet of
// VectorSamples given a slice of inputs and a
// slice of corresponding outputs.
func VectorSampleSet(inputs []linalg.Vector, outputs []linalg.Vector) sgd.SampleSet {
	if len(inputs) != len(outputs) {
		panic("input and output counts do not match")
	}
	res := make(sgd.SliceSampleSet, len(inputs))
	for i, in := range inputs {
		res[i] = VectorSample{Input: in, Output: outputs[i]}
	}
	return res
}
