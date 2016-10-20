package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A Block is a differentiable unit in a recurrent model
// that transforms input/state pairs into output/state
// pairs.
type Block interface {
	// StartState returns the initial state for the Block.
	StartState() State

	// StartStateR is like StartState but for an RState.
	StartRState(rv autofunc.RVector) RState

	// PropagateStart performs back-propagation through the
	// start state.
	// It takes the list of start states, their upstream
	// gradients, and the output gradient.
	PropagateStart(s []State, u []StateGrad, g autofunc.Gradient)

	// PropagateStartR is like PropagateStart, but for an
	// RStateGrad.
	// The g argument may be nil.
	PropagateStartR(s []RState, u []RStateGrad, rg autofunc.RGradient, g autofunc.Gradient)

	// ApplyBlock applies the block to a batch of inputs.
	// The result is valid so long as neither the inputs
	// nor the Block are changed.
	ApplyBlock(s []State, in []autofunc.Result) BlockResult

	// ApplyBlockR is like ApplyBlock, but with support for
	// the R operator.
	//
	// It is necessary to provide an RVector so that the
	// block knows how much each of its hidden parameters
	// changes with respect to R.
	ApplyBlockR(v autofunc.RVector, s []RState, in []autofunc.RResult) BlockRResult
}

// A BlockResult represents the output of a Block.
type BlockResult interface {
	// Outputs returns the vector outputs of the block.
	Outputs() []linalg.Vector

	// States returns the new states of the block.
	States() []State

	// PropagateGradient performs back-propagation.
	// It returns the gradient with respect to the input
	// states.
	//
	// A nil argument stands for a 0 gradient.
	// Upstream, s, and/or some entries in s may be nil.
	//
	// This should not modify the upstream information.
	PropagateGradient(upstream []linalg.Vector, s []StateGrad, g autofunc.Gradient) []StateGrad
}

// A BlockRResult is like a BlockResult, but for RStates.
type BlockRResult interface {
	// Outputs returns the vector outputs of the block.
	Outputs() []linalg.Vector

	// ROutputs returns the derivative of Outputs() with
	// respect to R.
	ROutputs() []linalg.Vector

	// RStates returns the new states of the block.
	RStates() []RState

	// PropagateRGradient performs back-propagation.
	// It returns the gradient with respect to the input
	// states.
	//
	// A nil upstream or s stands for a 0 gradient.
	// Upstream, s, and/or some entries in s may be nil.
	// The upstream argument may be nil if and only if the
	// upstreamR argument is also nil.
	// The g argument may also be nil if the gradients are
	// not desired.
	//
	// This should not modify the upstream information.
	PropagateRGradient(upstream, upstreamR []linalg.Vector, s []RStateGrad,
		rg autofunc.RGradient, g autofunc.Gradient) []RStateGrad
}
