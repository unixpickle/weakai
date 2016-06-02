package rnn

import "github.com/unixpickle/num-analysis/linalg"

// Gradient is a generic vector which can be
// added to a given RNN to adjust the RNN's
// weights, biases, and other parameters.
//
// Gradient implementations must store partials
// in zero or more vectors (i.e. float slices).
// This aggregate of vectors comprises a bigger
// hypothetical vector which is the "gradient".
type Gradient interface {
	// Inputs returns the partials of the cost
	// function with respect to the inputs of the
	// RNN that generated this Gradient.
	//
	// The Gradient should expect for the returned
	// vectors to be modified externally, but not
	// the slices containing the vectors.
	//
	// This is useful for stacking multiple RNNs
	// on top of each other and propagating gradients
	// through the aggregate.
	Inputs() []linalg.Vector

	// Params returns the partials of the cost
	// function with respect to the parameters of
	// the RNN that generated this Gradient.
	//
	// The Gradient should expect for the returned
	// vectors to be modified externally, but not
	// the slices containing the vectors.
	//
	// The returned slice must be structured the same
	// for all Gradients from a given RNN.
	// This way, multiple sets of param gradients can
	// be added up, etc.
	Params() []linalg.Vector
}

// AddGradients adds all of the components of g1 to g.
func AddGradients(g, g1 Gradient) {
	g1Inputs := g1.Inputs()
	for i, v := range g.Inputs() {
		v.Add(g1Inputs[i])
	}
	g1Params := g1.Params()
	for i, v := range g.Params() {
		v.Add(g1Params[i])
	}
}

// ScaleGradient scales the components of g.
func ScaleGradient(g Gradient, f float64) {
	for _, v := range g.Inputs() {
		v.Scale(f)
	}
	for _, v := range g.Params() {
		v.Scale(f)
	}
}
