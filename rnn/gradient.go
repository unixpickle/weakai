package rnn

import "github.com/unixpickle/num-analysis/linalg"

// Gradient is a generic vector which can be
// added to a given RNN to adjust the RNN's
// weights, biases, and other parameters.
type Gradient interface {
	// Inputs returns the partials of the cost
	// function with respect to the inputs of the
	// RNN that generated this Gradient.
	Inputs() []linalg.Vector

	// Scale scales the gradient in place and
	// returns the same gradient for convenience.
	Scale(f float64) Gradient

	// Add adds another gradient to this gradient
	// and returns this gradient for convenience.
	//
	// This is only guaranteed to work if both
	// gradients are from the same RNN.
	Add(g Gradient) Gradient

	// LargestComponent returns the largest partial
	// derivative (in terms of absolute value) out
	// of the partial derivatives for all parameters
	// of the underlying RNN.
	//
	// The exact meaning of this value may vary for
	// each Gradient implementation.
	//
	// This value should not count input partials.
	LargestComponent() float64

	// ClipComponents clips the magnitudes of partial
	// derivatives to a certain range.
	// If a partial derivative is greater than m or
	// less than -m, it will be set to m or -m
	// respectively.
	//
	// This should not affect input partials.
	ClipComponents(m float64)
}
