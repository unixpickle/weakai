package convnet

import "math"

// An ActivationFunc is a function designed to
// introduce non-linearity into a neural net.
type ActivationFunc interface {
	// Eval evaluates the activation function for
	// an input x.
	Eval(x float64) float64

	// Deriv evaluates the derivative of the
	// activation function for an input x.
	Deriv(x float64) float64
}

type Sigmoid struct{}

func (_ Sigmoid) Eval(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (s Sigmoid) Deriv(x float64) float64 {
	v := s.Eval(x)
	return v * (1.0 - v)
}
