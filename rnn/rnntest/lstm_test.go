package rnntest

import (
	"testing"

	"github.com/unixpickle/weakai/rnn"
)

func TestLSTMGradients(t *testing.T) {
	test := GradientTest{
		Block: rnn.StackedBlock{rnn.NewLSTM(3, 2),
			NewSquareBlock(2)},
		GradientParams: gradientTestVariables,
		Inputs:         gradientTestVariables[:2],
		InStates:       gradientTestVariables[6:8],
	}
	test.Run(t)
	test.GradientParams = nil
	test.Run(t)
}

func TestLSTMBatches(t *testing.T) {
	batchTest := BatchTest{
		Block: rnn.StackedBlock{rnn.NewLSTM(3, 2), NewSquareBlock(2)},

		OutputSize:     2,
		GradientParams: gradientTestVariables,
		Inputs:         gradientTestVariables[:2],
		InStates:       gradientTestVariables[6:8],
	}
	batchTest.Run(t)
	batchTest.GradientParams = nil
	batchTest.Run(t)
}

/*
func TestLSTMBatchOutputs(t *testing.T) {
	block := rnn.StackedBlock{rnn.NewLSTM(3, 5), NewSquareBlock(1)}
	inputs := gradientTestVariables[:2]
	inStates := gradientTestVariables[6:8]

	var realOutputs []linalg.Vector
	var realOutStates []linalg.Vector
	for i, in := range inputs {
		blockIn := &rnn.BlockInput{
			Inputs: []*autofunc.Variable{in},
			States: []*autofunc.Variable{inStates[i]},
		}
		output := block.Batch(blockIn)
		realOutputs = append(realOutputs, output.Outputs()[0])
		realOutStates = append(realOutStates, output.States()[0])
	}

	bigBlockIn := &rnn.BlockInput{
		Inputs: inputs,
		States: inStates,
	}
	bigOutput := block.Batch(bigBlockIn)

	for i, expOut := range realOutputs {
		actOut := bigOutput.Outputs()[i]
		expState := realOutStates[i]
		actState := bigOutput.States()[i]
		for i, x := range expOut {
			a := actOut[i]
			if math.Abs(x-a) > lstmTestPrec {
				t.Errorf("lane %d: expected output %f got %f", i, x, a)
			}
		}
		for i, x := range expState {
			a := actState[i]
			if math.Abs(x-a) > lstmTestPrec {
				t.Errorf("lane %d: expected state %f got %f", i, x, a)
			}
		}
	}
}

func TestLSTMBatchGradients(t *testing.T) {
	block := rnn.StackedBlock{rnn.NewLSTM(3, 5), NewSquareBlock(1)}
	inputs := gradientTestVariables[:2]
	inStates := gradientTestVariables[6:8]

	upstream := &rnn.UpstreamGradient{
		Outputs: []linalg.Vector{
			{1, -2, 3, 2, -1}, {1, -1, -2, 3, 2},
		},
		States: []linalg.Vector{
			{1, 2, -3, 2, 5, 3}, {5, -4, 3, -2, 2, 4},
		},
	}

	realGradient := autofunc.NewGradient(block.Parameters())
	for i, in := range inputs {
		blockIn := &rnn.BlockInput{
			Inputs: []*autofunc.Variable{in},
			States: []*autofunc.Variable{inStates[i]},
		}
		localUpstream := &rnn.UpstreamGradient{
			Outputs: upstream.Outputs[i : i+1],
			States:  upstream.States[i : i+1],
		}
		output := block.Batch(blockIn)
		output.Gradient(localUpstream, realGradient)
	}

	bigBlockIn := &rnn.BlockInput{
		Inputs: inputs,
		States: inStates,
	}
	bigOutput := block.Batch(bigBlockIn)
	bigGradient := autofunc.NewGradient(block.Parameters())
	bigOutput.Gradient(upstream, bigGradient)

	for variable, expVec := range realGradient {
		actVec := bigGradient[variable]
		for i, x := range expVec {
			a := actVec[i]
			if math.Abs(x-a) > lstmTestPrec {
				t.Errorf("got invalid gradient output %f (expected %f)", a, x)
			}
		}
	}
}
*/
