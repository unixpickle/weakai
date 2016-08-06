package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A SeqFunc is a function that can be applied to
// variable-length input sequences.
type SeqFunc interface {
	// BatchSeqs applies the function to each of the
	// sequences in seqs, where a sequence is stored
	// as a slice of vectors (autofunc.Results).
	// The output sequences will be in the same order
	// as the input sequences.
	BatchSeqs(seqs [][]autofunc.Result) ResultSeqs

	// BatchSeqsR is like BatchSeqs but with support
	// for the R-operator.
	BatchSeqsR(rv autofunc.RVector, seqs [][]autofunc.RResult) RResultSeqs
}

// ResultSeqs is the output of a SeqFunc, storing a
// batch of output sequences and capable of
// back-propagation.
type ResultSeqs interface {
	// OutputSeqs is a slice of sequences, where each
	// sequence is a slice of output vectors.
	OutputSeqs() [][]linalg.Vector

	// Gradient performs back-propagation through the
	// output sequence, given the rates of changes of
	// all of the output sequences.
	// The appropriate gradients are added to g.
	Gradient(upstream [][]linalg.Vector, g autofunc.Gradient)
}

// RResultSeqs is like OutputSeqs but with extra
// R-operator information.
type RResultSeqs interface {
	OutputSeqs() [][]linalg.Vector
	ROutputSeqs() [][]linalg.Vector

	// RGradient performs back-propagation.
	// The gradient g may be nil.
	RGradient(upstream, upstreamR [][]linalg.Vector,
		rg autofunc.RGradient, g autofunc.Gradient)
}
