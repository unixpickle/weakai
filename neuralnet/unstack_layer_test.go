package neuralnet

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

func TestUnstackLayerInverse(t *testing.T) {
	// This test utilizes the fact that gradients have to
	// be un-unstacked when they are propagated backwards.
	// Thus, we can check that unstacking and un-unstacking
	// are truly inverses as they should be.

	width := 15
	height := 13
	depth := 18

	inputVal := make(linalg.Vector, width*height*depth)
	for i := range inputVal {
		inputVal[i] = rand.Float64()*2 - 1
	}
	variable := &autofunc.Variable{inputVal}

	layer := &UnstackLayer{
		InputWidth:    width,
		InputHeight:   height,
		InputDepth:    depth,
		InverseStride: 3,
	}
	output := layer.Apply(variable)

	outGrad := make(linalg.Vector, len(output.Output()))
	copy(outGrad, output.Output())
	grad := autofunc.NewGradient([]*autofunc.Variable{variable})
	output.PropagateGradient(outGrad, grad)

	original := grad[variable]
	if len(original) != len(inputVal) {
		t.Fatalf("expected output length %d got %d", len(inputVal), len(original))
	}

	for i, x := range inputVal {
		a := original[i]
		if math.Abs(a-x) > 1e-6 {
			t.Fatalf("entry %d should be %f but got %f", i, x, a)
		}
	}
}

func TestUnstackLayerRProp(t *testing.T) {
	width := 3
	height := 4
	depth := 18

	inputVal := make(linalg.Vector, width*height*depth)
	inputR := make(linalg.Vector, len(inputVal))
	for i := range inputVal {
		inputVal[i] = rand.Float64()*2 - 1
		inputR[i] = rand.Float64()*2 - 1
	}
	variable := &autofunc.Variable{inputVal}
	rVec := autofunc.RVector{
		variable: inputR,
	}

	layer := &UnstackLayer{
		InputWidth:    width,
		InputHeight:   height,
		InputDepth:    depth,
		InverseStride: 3,
	}

	funcTest := &functest.RFuncTest{
		F:     layer,
		Vars:  []*autofunc.Variable{variable},
		Input: variable,
		RV:    rVec,
	}
	funcTest.Run(t)
}
