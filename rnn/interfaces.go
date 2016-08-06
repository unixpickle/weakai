// Package rnn facilitates the evaluation and training
// of recurrent neural networks.
package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// UpstreamGradient stores the gradients of some
// output with respect to the outputs and output
// states of some Block.
// Either one of the slices (States or Outputs)
// may be nil, indicating that said gradient is
// completely 0.
type UpstreamGradient struct {
	States  []linalg.Vector
	Outputs []linalg.Vector
}

// UpstreamRGradient is like UpstreamGradient,
// but it stores the derivatives of all the
// partials with respect to some variable R.
//
// A slice (States or Outputs) can be nil if and
// only if its corresponding R slice is also nil.
type UpstreamRGradient struct {
	UpstreamGradient
	RStates  []linalg.Vector
	ROutputs []linalg.Vector
}

// A BlockInput stores a batch of states and inputs
// for a Block.
type BlockInput struct {
	States []*autofunc.Variable
	Inputs []*autofunc.Variable
}

// A BlockOutput represents a batch of outputs and new
// states from a Block.
type BlockOutput interface {
	States() []linalg.Vector
	Outputs() []linalg.Vector

	// Gradient updates the gradients in g given the
	// upstream gradient from this BlockOutput.
	// This should not modify u.
	Gradient(u *UpstreamGradient, g autofunc.Gradient)
}

// A BlockRInput is like a BlockInput, but includes
// derivatives of all the inputs and states with
// respect to some variable R.
type BlockRInput struct {
	States []*autofunc.RVariable
	Inputs []*autofunc.RVariable
}

// An BlockROutput is like a BlockOutput, but includes
// derivatives of the outputs and states with respect
// to some variable R.
type BlockROutput interface {
	States() []linalg.Vector
	Outputs() []linalg.Vector
	RStates() []linalg.Vector
	ROutputs() []linalg.Vector

	// RGradient updates the gradients in g and the
	// r-gradients in rg given the upstream gradient
	// u and the derivative of u with respect to R,
	// stored in ru.
	// The gradient g may be nil to indicate that only
	// the r-gradient is desired.
	// This should not modify u.
	RGradient(u *UpstreamRGradient, rg autofunc.RGradient, g autofunc.Gradient)
}

// A Block is a unit in a Recurrent Neural Network that
// transforms input-state pairs into output/state pairs.
type Block interface {
	// StateSize returns the number of values in each
	// state of the Block.
	StateSize() int

	// Batch applies forward propagation to a BlockInput.
	// The result is valid so long as neither the input
	// nor the Block is changed.
	Batch(in *BlockInput) BlockOutput

	// BatchR is like Batch, but for an BlockRInput.
	// The result is valid so long as neither the input
	// nor the Block is changed.
	//
	// It is necessary to provide an RVector so that the
	// block knows how much each of its hidden parameters
	// changes with respect to R.
	BatchR(v autofunc.RVector, in *BlockRInput) BlockROutput
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
