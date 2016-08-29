package rnntest

import (
	"testing"

	"github.com/unixpickle/weakai/rnn"
)

func TestStateOutBlockGradients(t *testing.T) {
	test := GradientTest{
		Block: &rnn.StateOutBlock{
			Block: NewDemoBlock(3, 2, 3),
		},
		GradientParams: gradientTestVariables,
		Inputs:         gradientTestVariables[:2],
		InStates:       gradientTestVariables[2:4],
	}
	test.Run(t)
	test.GradientParams = nil
	test.Run(t)
}
