package neuralnet

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

type meanSquaredTestFunc struct {
	Expected linalg.Vector
}

func (m meanSquaredTestFunc) Apply(in autofunc.Result) autofunc.Result {
	return MeanSquaredCost{}.Cost(m.Expected, in)
}

func (m meanSquaredTestFunc) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return MeanSquaredCost{}.CostR(v, m.Expected, in)
}

func TestMeanSquaredCostGradient(t *testing.T) {
	actual := &autofunc.Variable{make(linalg.Vector, 10)}
	expected := make(linalg.Vector, len(actual.Vector))
	for i := range expected {
		expected[i] = rand.Float64()
		actual.Vector[i] = rand.Float64()
	}
	funcTest := &functest.FuncTest{
		F:     meanSquaredTestFunc{expected},
		Vars:  []*autofunc.Variable{actual},
		Input: actual,
	}
	funcTest.Run(t)
}

func TestMeanSquaredCostRGradient(t *testing.T) {
	actual := &autofunc.Variable{make(linalg.Vector, 10)}
	expected := make(linalg.Vector, len(actual.Vector))
	rVector := autofunc.RVector{actual: make(linalg.Vector, len(expected))}
	for i := range expected {
		expected[i] = rand.Float64()
		actual.Vector[i] = rand.Float64()
		rVector[actual][i] = rand.Float64()
	}
	funcTest := &functest.RFuncTest{
		F:     meanSquaredTestFunc{expected},
		Vars:  []*autofunc.Variable{actual},
		Input: autofunc.NewRVariable(actual, rVector),
		RV:    rVector,
	}
	funcTest.Run(t)
}
