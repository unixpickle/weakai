package rnntest

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

const batchTestPrec = 1e-3

type BatchTest struct {
	Block          BlockLearner
	OutputSize     int
	GradientParams []*autofunc.Variable
	Inputs         []*autofunc.Variable
	InStates       []*autofunc.Variable
}

func (b *BatchTest) Run(t *testing.T) {
	b.checkOutput(t)
	b.checkGradient(t)
}

func (b *BatchTest) checkOutput(t *testing.T) {
	var realOutputs []linalg.Vector
	var realOutStates []linalg.Vector
	for i, in := range b.Inputs {
		blockIn := &rnn.BlockInput{
			Inputs: []*autofunc.Variable{in},
			States: []*autofunc.Variable{b.InStates[i]},
		}
		output := b.Block.Batch(blockIn)
		realOutputs = append(realOutputs, output.Outputs()[0])
		realOutStates = append(realOutStates, output.States()[0])
	}

	bigBlockIn := &rnn.BlockInput{
		Inputs: b.Inputs,
		States: b.InStates,
	}
	bigOutput := b.Block.Batch(bigBlockIn)

	for i, expOut := range realOutputs {
		actOut := bigOutput.Outputs()[i]
		expState := realOutStates[i]
		actState := bigOutput.States()[i]
		for i, x := range expOut {
			a := actOut[i]
			if math.Abs(x-a) > batchTestPrec {
				t.Errorf("lane %d: expected output %f got %f", i, x, a)
				break
			}
		}
		for i, x := range expState {
			a := actState[i]
			if math.Abs(x-a) > batchTestPrec {
				t.Errorf("lane %d: expected state %f got %f", i, x, a)
				break
			}
		}
	}
}

func (b *BatchTest) checkGradient(t *testing.T) {
	upstream := &rnn.UpstreamGradient{}
	for _ = range b.Inputs {
		outVec := make(linalg.Vector, b.OutputSize)
		for i := range outVec {
			outVec[i] = rand.NormFloat64()
		}
		stateVec := make(linalg.Vector, b.Block.StateSize())
		for i := range stateVec {
			stateVec[i] = rand.NormFloat64()
		}
		upstream.Outputs = append(upstream.Outputs, outVec)
		upstream.States = append(upstream.States, stateVec)
	}

	realGradient := autofunc.NewGradient(b.allParams())
	for i, in := range b.Inputs {
		blockIn := &rnn.BlockInput{
			Inputs: []*autofunc.Variable{in},
			States: []*autofunc.Variable{b.InStates[i]},
		}
		localUpstream := &rnn.UpstreamGradient{
			Outputs: upstream.Outputs[i : i+1],
			States:  upstream.States[i : i+1],
		}
		output := b.Block.Batch(blockIn)
		output.Gradient(localUpstream, realGradient)
	}

	bigBlockIn := &rnn.BlockInput{
		Inputs: b.Inputs,
		States: b.InStates,
	}
	bigOutput := b.Block.Batch(bigBlockIn)
	bigGradient := autofunc.NewGradient(b.allParams())
	bigOutput.Gradient(upstream, bigGradient)

	for variable, expVec := range realGradient {
		actVec := bigGradient[variable]
		for i, x := range expVec {
			a := actVec[i]
			if math.Abs(x-a) > batchTestPrec {
				t.Errorf("got invalid gradient output %f (expected %f)", a, x)
				return
			}
		}
	}
}

func (b *BatchTest) allParams() []*autofunc.Variable {
	params := make([]*autofunc.Variable, len(b.GradientParams))
	copy(params, b.GradientParams)
	params = append(params, b.Block.Parameters()...)
	return params
}
