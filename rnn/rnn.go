package rnn

import "github.com/unixpickle/num-analysis/linalg"

// An RNN is a generic Recurrent Neural Network.
type RNN interface {
	// Randomize randomly initializes the parameters
	// of the RNN.
	Randomize()

	// StepTime runs the next time step and returns
	// the output computed by the RNN given existing
	// state and the new input.
	StepTime(input linalg.Vector) linalg.Vector

	// CostGradient computes the gradient of the cost
	// function for this RNN's parameters.
	//
	// The caller must pass an array of output gradients,
	// where each output gradient corresponds to a time
	// step, and each element of each vector corresponds
	// to the partial of the cost function with respect
	// to that output of the RNN at that time.
	//
	// The number of elements in outGradients should match
	// the number of StepTime() calls that were made since
	// the RNN was created or reset.
	CostGradient(outGradients []linalg.Vector) Gradient

	// Reset resets the state of the RNN, essentially
	// clearing the RNN's memory of previous states.
	Reset()

	// StepGradient updates the RNN's parameters by adding
	// the gradient to them.
	StepGradient(grad Gradient)
}
