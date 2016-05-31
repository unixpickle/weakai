package lstm

import "math"

type ActivationFunc interface {
	// Eval evaluates the activation function for
	// the given input.
	Eval(x float64) float64

	// Deriv evaluates the activation function's
	// derivative, given its output from Eval().
	// This means that activation functions'
	// derivatives must be relatively easy to
	// compute given the functions' values.
	// For example, the derivative of Sigmoid(x)
	// where Sigmoid(x)=S is S*(1-S).
	Deriv(evalOut float64) float64
}

type Sigmoid struct{}

func (_ Sigmoid) Eval(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (_ Sigmoid) Deriv(evalOut float64) float64 {
	return evalOut * (1 - evalOut)
}

type ReLU struct{}

func (_ ReLU) Eval(x float64) float64 {
	if x < 0 {
		return 0
	} else {
		return x
	}
}

func (_ ReLU) Deriv(evalOut float64) float64 {
	if evalOut > 0 {
		return 1
	} else {
		return 0
	}
}

type Tanh struct{}

func (_ Tanh) Eval(x float64) float64 {
	return math.Tanh(x)
}

func (_ Tanh) Deriv(x float64) float64 {
	return 1 - x*x
}
