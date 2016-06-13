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
	Gradient(g autofunc.Gradient, u *UpstreamGradient)
}

// An RBlockInput is like a BlockInput, but includes
// derivatives of all the inputs and states with
// respect to some variable R.
type RBlockInput struct {
	BlockInput
	RStates []linalg.Vector
	RInputs []linalg.Vector
}

// An RBlockOutput is like a BlockOutput, but includes
// derivatives of the outputs and states with respect
// to some variable R.
type RBlockOutput interface {
	BlockOutput
	RStates() []linalg.Vector
	ROutputs() []linalg.Vector

	// RGradient updates the gradients in g and the
	// r-gradients in rg given the upstream gradient
	// u and the derivative of u with respect to R,
	// stored in ru.
	RGradient(g autofunc.Gradient, rg autofunc.RGradient, u, ru *UpstreamGradient)
}

// A Block is a unit in a Recurrent Neural Network that
// transforms input-state pairs into output/state pairs.
type Block interface {
	// Forward applies forward propagation to a BlockInput.
	Batch(in BlockInput) BlockOutput

	// ForwardR is like Forward, but for an RBlockInput.
	// It is necessary to provide an RVector so that the
	// block knows how much each of its hidden parameters
	// changes with respect to R.
	BatchR(v autofunc.RVector, in RBlockInput) RBlockOutput
}

// A Learner has parameters which can be trained using
// some variant of gradient descent.
type Learner interface {
	Parameters() []*autofunc.Variable
}
