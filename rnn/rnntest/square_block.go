package rnntest

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/weakai/rnn"
)

// NewSquareBlock creates a block which squares
// its input values and states.
// This is useful for testing r-gradient propagation,
// since the derivative of x^2 depends on x, so if
// x changes with r, so does the derivative of x^2.
func NewSquareBlock(stateSize int) rnn.Block {
	return &rnn.BatcherBlock{
		F:            &autofunc.RFuncBatcher{F: squareFunc{}},
		StateSizeVal: stateSize,
	}
}

type squareFunc struct{}

func (_ squareFunc) Apply(in autofunc.Result) autofunc.Result {
	return autofunc.Square(in)
}

func (_ squareFunc) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return autofunc.SquareR(in)
}
