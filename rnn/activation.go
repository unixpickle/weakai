package rnn

import (
	"math"

	"github.com/unixpickle/serializer"
)

// An ActivationFunc is used to "squash" output
// from neurons.
type ActivationFunc interface {
	serializer.Serializer

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

func (_ Sigmoid) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (_ Sigmoid) SerializerType() string {
	return serializerTypeSigmoid
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

func (_ ReLU) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (_ ReLU) SerializerType() string {
	return serializerTypeReLU
}

type Tanh struct{}

func (_ Tanh) Eval(x float64) float64 {
	return math.Tanh(x)
}

func (_ Tanh) Deriv(x float64) float64 {
	return 1 - x*x
}

func (_ Tanh) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (_ Tanh) SerializerType() string {
	return serializerTypeTanh
}
