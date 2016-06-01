package neuralnet

import "github.com/unixpickle/serializer"

// A Layer represents any kind of "layer"
// that a deep neural network may contain.
//
// A Layer is capable of propagating values
// from a set of inputs to a set of outputs,
// performing some function along the way.
//
// Layers are also capable of differentiating
// the function they perform, doing what is
// known as back propagation.
type Layer interface {
	serializer.Serializer

	// Output is the output vector for this layer.
	// The contents of this slice may change after
	// forward propagation, but the slice itself
	// should remain the same, such that Output()
	// need only be called once.
	Output() []float64

	// UpstreamGradient is set by back propagation
	// to indicate the rate of change of the cost
	// function with respect to each of the inputs
	// to this layer.
	// The length of this vector must match the
	// length of Input().
	UpstreamGradient() []float64

	// Input returns the current input vector, as set
	// by SetInput().
	Input() []float64

	// SetInput tells this layer where to get its
	// input for forward propagation.
	// It returns false if the input's length is
	// incorrect.
	SetInput([]float64) bool

	// DownstreamGradient returns the current downstream
	// gradient, as set by SetDownstreamGradient().
	DownstreamGradient() []float64

	// SetDownstreamGradient sets the vector which,
	// during back propagation, specifies the rate of
	// change of the cost function with respect to
	// each of the outputs from this layer.
	// It returns false if the input's length is
	// incorrect.
	SetDownstreamGradient([]float64) bool

	// Randomize randomly adjusts the parameters of
	// this layer.
	Randomize()

	// PropagateForward performs forward propagation,
	// using the inputs to generate outputs.
	PropagateForward()

	// PropagateBackward performs backward propagation,
	// using the downstream gradients and information
	// from the latest PropagateForward() to generate
	// internal and, potentially, upstream gradients.
	//
	// If upstream is false, then this will only
	// generate internal gradients, not upstream ones.
	PropagateBackward(upstream bool)

	// GradientMagSquared takes the gradient vector of
	// this layer's parameters with respect to the cost
	// function, computes its magnitude, and returns
	// the magnitude's square.
	//
	// It uses the gradient computed during the last
	// back propagation.
	GradientMagSquared() float64

	// StepGradient adjusts the layer's parameters
	// by adding f times the parameter gradient to the
	// parameters.
	//
	// It uses the gradient computed during the last
	// back propagation.
	StepGradient(f float64)

	// Alias creates another version of this Layer
	// which shares the layer's parameters but may
	// have different inputs, outputs, and gradients.
	//
	// Multiple aliases of a layer may do propagation
	// simultaneously, but only one alias may run
	// StepGradient() at once, since it changes the
	// layer's parameters.
	Alias() Layer
}

// LayerPrototype represents a prototype
// with which new Layers can be made.
type LayerPrototype interface {
	Make() Layer
}
