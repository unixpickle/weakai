package rnntest

import (
	"testing"

	"github.com/unixpickle/weakai/rnn"
)

func TestLSTMGradients(t *testing.T) {
	test := GradientTest{
		Block: rnn.StackedBlock{rnn.NewLSTM(3, 5),
			NewSquareBlock(1)},
		GradientParams: gradientTestVariables,
		Inputs:         gradientTestVariables[:2],
		InStates:       gradientTestVariables[6:8],
	}
	test.Run(t)
}
