// Package seqtoseq implements gradient-based training
// for models which take an input sequence and produce
// an output sequence of the same length.
package seqtoseq

import "github.com/unixpickle/num-analysis/linalg"

// Sample is a training sample containing an input
// sequence and its corresponding output sequence.
// All gradienters in this package take sample sets
// with Sample elements.
type Sample struct {
	Inputs  []linalg.Vector
	Outputs []linalg.Vector
}
