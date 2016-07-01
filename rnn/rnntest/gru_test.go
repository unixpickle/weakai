package rnntest

import (
	"testing"

	"github.com/unixpickle/weakai/rnn"
)

func TestGRUGradients(t *testing.T) {
	test := GradientTest{
		Block: rnn.StackedBlock{rnn.NewGRU(3, 4),
			NewSquareBlock(2)},
		GradientParams: gradientTestVariables,
		Inputs:         gradientTestVariables[:2],
		InStates:       gradientTestVariables[6:8],
	}
	test.Run(t)
	test.GradientParams = nil
	test.Run(t)
	test.GradientParams = gradientTestVariables
	test.Block = rnn.NewGRU(3, 6)
	test.Run(t)
	test.GradientParams = nil
	test.Run(t)
}

func TestGRUBatches(t *testing.T) {
	batchTest := BatchTest{
		Block: rnn.StackedBlock{rnn.NewGRU(3, 4), NewSquareBlock(2)},

		OutputSize:     4,
		GradientParams: gradientTestVariables,
		Inputs:         gradientTestVariables[:2],
		InStates:       gradientTestVariables[6:8],
	}
	batchTest.Run(t)
	batchTest.GradientParams = nil
	batchTest.Run(t)
}
