package rnntest

import (
	"testing"

	"github.com/unixpickle/autofunc"
)

var gradientTestVariables = []*autofunc.Variable{
	&autofunc.Variable{
		Vector: []float64{1, 2, 3},
	},
	&autofunc.Variable{
		Vector: []float64{1, -2, 3},
	},
	&autofunc.Variable{
		Vector: []float64{3, -4},
	},
	&autofunc.Variable{
		Vector: []float64{3, 4},
	},
}

func TestGradientTestIdentity(t *testing.T) {
	test := GradientTest{
		Block:          IdentityBlock{StateSizeVal: 2},
		GradientParams: gradientTestVariables,
		Inputs:         gradientTestVariables[:2],
		InStates:       gradientTestVariables[2:],
	}
	test.Run(t)
}
