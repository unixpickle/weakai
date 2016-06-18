package rnntest

import (
	"testing"

	"github.com/unixpickle/weakai/rnn"
)

func TestStackedBlockIdentity(t *testing.T) {
	test := GradientTest{
		Block:          rnn.StackedBlock{IdentityBlock{StateSizeVal: 2}},
		GradientParams: gradientTestVariables,
		Inputs:         gradientTestVariables[:2],
		InStates:       gradientTestVariables[2:4],
	}
	test.Run(t)
	test.GradientParams = nil
	test.Run(t)
}

func TestStackedBlockSquare(t *testing.T) {
	test := GradientTest{
		Block: rnn.StackedBlock{IdentityBlock{StateSizeVal: 0},
			NewSquareBlock(2)},
		GradientParams: gradientTestVariables,
		Inputs:         gradientTestVariables[:2],
		InStates:       gradientTestVariables[2:4],
	}
	test.Run(t)
	test.GradientParams = nil
	test.Run(t)
}

func TestStackedBlockDeepDemo(t *testing.T) {
	test := GradientTest{
		Block: rnn.StackedBlock{NewDemoBlock(3, 2, 3),
			NewSquareBlock(2)},
		GradientParams: gradientTestVariables,
		Inputs:         gradientTestVariables[:2],
		InStates:       gradientTestVariables[4:6],
	}
	test.Run(t)
	test.GradientParams = nil
	test.Run(t)
}

func TestStackedBlockNested(t *testing.T) {
	test := GradientTest{
		Block: rnn.StackedBlock{rnn.StackedBlock{NewDemoBlock(3, 2, 3),
			NewSquareBlock(1)}, rnn.StackedBlock{NewDemoBlock(3, 1, 5),
			NewSquareBlock(2)}},
		GradientParams: gradientTestVariables,
		Inputs:         gradientTestVariables[:2],
		InStates:       gradientTestVariables[6:8],
	}
	test.Run(t)
	test.GradientParams = nil
	test.Run(t)
}
