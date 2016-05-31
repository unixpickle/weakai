package rnn

import (
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

// An RNN is a generic Recurrent Neural Network.
type RNN interface {
	serializer.Serializer

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

	// Alias creates a read-only alias of this RNN.
	// All aliases of an RNN share parameters (e.g. weights)
	// but do not share memory.
	//
	// Each alias can run StepTime(), CostGradient(), and
	// Reset() calls concurrently from every other alias,
	// but only one Alias can run StepGradient() or Randomize()
	// at once.
	Alias() RNN
}