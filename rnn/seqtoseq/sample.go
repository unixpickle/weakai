// Package seqtoseq implements gradient-based training
// for models which take an input sequence and produce
// an output sequence of the same length.
package seqtoseq

import (
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
)

// Sample is a training sample containing an input
// sequence and its corresponding output sequence.
// All gradienters in this package take sample sets
// with Sample elements.
type Sample struct {
	Inputs  []linalg.Vector
	Outputs []linalg.Vector
}

// Hash returns a randomly-distributed hash of the sample.
func (s Sample) Hash() []byte {
	allVecs := make([]linalg.Vector, len(s.Inputs)+len(s.Outputs))
	copy(allVecs, s.Inputs)
	copy(allVecs[len(s.Inputs):], s.Outputs)
	return sgd.HashVectors(allVecs...)
}
