package rnntest

import (
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

// DemoBlock is an rnn.Block which performs a randomly
// generated linear transformation on inputs.
type DemoBlock struct {
	Matrix       *autofunc.LinTran
	StateSizeVal int
}

// NewDemoBlock creates a new randomized DemoBlock with
// the given dimensions.
func NewDemoBlock(inSize, stateSize, outSize int) *DemoBlock {
	matData := make(linalg.Vector, (inSize+stateSize)*(outSize+stateSize))
	for i := range matData {
		matData[i] = rand.NormFloat64()
	}
	return &DemoBlock{
		Matrix: &autofunc.LinTran{
			Rows: outSize + stateSize,
			Cols: inSize + stateSize,
			Data: &autofunc.Variable{Vector: matData},
		},
		StateSizeVal: stateSize,
	}
}

func (d *DemoBlock) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{d.Matrix.Data}
}

func (d *DemoBlock) StateSize() int {
	return d.StateSizeVal
}

func (d *DemoBlock) Batch(in *rnn.BlockInput) rnn.BlockOutput {
	b := &rnn.BatcherBlock{
		F:            d.Matrix,
		StateSizeVal: d.StateSizeVal,
	}
	return b.Batch(in)
}

func (d *DemoBlock) BatchR(v autofunc.RVector, in *rnn.BlockRInput) rnn.BlockROutput {
	b := &rnn.BatcherBlock{
		F:            d.Matrix,
		StateSizeVal: d.StateSizeVal,
	}
	return b.BatchR(v, in)
}
