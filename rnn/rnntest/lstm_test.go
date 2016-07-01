package rnntest

import (
	"testing"

	"github.com/unixpickle/weakai/rnn"
)

const lstmTestPrec = 1e-5

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
