package neuralnet

import (
	"fmt"
	"math"

	"github.com/unixpickle/serializer"
)

// An ActivationFunc is a function designed to
// introduce non-linearity into a neural net.
type ActivationFunc interface {
	serializer.Serializer

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

func (_ Sigmoid) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (_ Sigmoid) SerializerType() string {
	return serializerTypeSigmoid
}

type HyperbolicTangent struct{}

func (_ HyperbolicTangent) Eval(x float64) float64 {
	return math.Tanh(x)
}

func (_ HyperbolicTangent) Deriv(x float64) float64 {
	cosh := math.Cosh(x)
	if math.IsInf(cosh, 0) {
		return 0
	}
	coshSquared := math.Pow(cosh, 2)
	if math.IsInf(coshSquared, 0) {
		return 0
	}
	return 1 / coshSquared
}

func (_ HyperbolicTangent) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (_ HyperbolicTangent) SerializerType() string {
	return serializerTypeHyperbolicTangent
}

func deserializeActivation(data []byte) (ActivationFunc, error) {
	activation, err := serializer.DeserializeWithType(data)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize activation: %s", err.Error())
	} else if rightType, ok := activation.(ActivationFunc); !ok {
		return nil, fmt.Errorf("expected ActivationFunc but got %T", activation)
	} else {
		return rightType, nil
	}
}
