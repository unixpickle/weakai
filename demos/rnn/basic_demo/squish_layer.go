package main

import (
	"errors"

	"github.com/unixpickle/autofunc"
)

const squishAmount = 0.999

type SquishLayer struct{}

func (_ SquishLayer) Apply(in autofunc.Result) autofunc.Result {
	return autofunc.AddScaler(autofunc.Scale(in, squishAmount), (1-squishAmount)/2)
}

func (_ SquishLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return autofunc.AddScalerR(autofunc.ScaleR(in, squishAmount), (1-squishAmount)/2)
}

func (_ SquishLayer) Serialize() ([]byte, error) {
	return nil, errors.New("not implemented")
}

func (_ SquishLayer) SerializerType() string {
	return ""
}
