package neuralnet

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type batchFunc interface {
	autofunc.Batcher
	autofunc.Func
}

type batchFuncR interface {
	autofunc.RBatcher
	autofunc.RFunc
}

func testBatcher(t *testing.T, b batchFunc, in autofunc.Result, n int,
	params []*autofunc.Variable) {
	funcBatcher := autofunc.FuncBatcher{F: b}

	t.Run("Forward", func(t *testing.T) {
		expected := funcBatcher.Batch(in, n)
		actual := b.Batch(in, n)
		diff := actual.Output().Copy().Scale(-1).Add(expected.Output()).MaxAbs()
		if diff > 1e-5 {
			t.Errorf("expected output %v but got %v", expected, actual)
		}
	})

	t.Run("Backward", func(t *testing.T) {
		expectedOut := funcBatcher.Batch(in, n)
		actualOut := b.Batch(in, n)

		expected := autofunc.NewGradient(params)
		actual := autofunc.NewGradient(params)

		outGrad := make(linalg.Vector, len(expectedOut.Output()))
		for i := range outGrad {
			outGrad[i] = rand.NormFloat64()
		}

		expectedOut.PropagateGradient(outGrad.Copy(), expected)
		actualOut.PropagateGradient(outGrad, actual)

		for i, variable := range params {
			actualVec := actual[variable]
			expectedVec := expected[variable]
			diff := actualVec.Copy().Scale(-1).Add(expectedVec).MaxAbs()
			if diff > 1e-5 {
				t.Errorf("variable %d: expected %v got %v", i, expectedVec, actualVec)
			}
		}
	})
}

func testRBatcher(t *testing.T, rv autofunc.RVector, b batchFuncR, in autofunc.RResult,
	n int, params []*autofunc.Variable) {
	funcRBatcher := autofunc.RFuncBatcher{F: b}

	t.Run("Forward", func(t *testing.T) {
		expected := funcRBatcher.BatchR(rv, in, n)
		actual := b.BatchR(rv, in, n)
		diff := actual.Output().Copy().Scale(-1).Add(expected.Output()).MaxAbs()
		if diff > 1e-5 {
			t.Errorf("expected output %v but got %v", expected, actual)
		}
		diff = actual.ROutput().Copy().Scale(-1).Add(expected.ROutput()).MaxAbs()
		if diff > 1e-5 {
			t.Errorf("expected r-output %v but got %v", expected, actual)
		}
	})

	t.Run("Backward", func(t *testing.T) {
		expectedOut := funcRBatcher.BatchR(rv, in, n)
		actualOut := b.BatchR(rv, in, n)

		expected := autofunc.NewGradient(params)
		actual := autofunc.NewGradient(params)
		expectedR := autofunc.NewRGradient(params)
		actualR := autofunc.NewRGradient(params)

		outGrad := make(linalg.Vector, len(expectedOut.Output()))
		outGradR := make(linalg.Vector, len(expectedOut.Output()))
		for i := range outGrad {
			outGrad[i] = rand.NormFloat64()
			outGradR[i] = rand.NormFloat64()
		}

		expectedOut.PropagateRGradient(outGrad.Copy(), outGradR.Copy(), expectedR, expected)
		actualOut.PropagateRGradient(outGrad, outGradR, actualR, actual)

		for i, variable := range params {
			actualVec := actual[variable]
			expectedVec := expected[variable]
			diff := actualVec.Copy().Scale(-1).Add(expectedVec).MaxAbs()
			if diff > 1e-5 {
				t.Errorf("variable %d (grad): expected %v got %v", i, expectedVec, actualVec)
			}
			actualVec = actualR[variable]
			expectedVec = expectedR[variable]
			diff = actualVec.Copy().Scale(-1).Add(expectedVec).MaxAbs()
			if diff > 1e-5 {
				t.Errorf("variable %d (rgrad): expected %v got %v", i, expectedVec, actualVec)
			}
		}
	})
}
