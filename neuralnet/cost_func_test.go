package neuralnet

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
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
	funcTest := &functest.FuncChecker{
		F:     meanSquaredTestFunc{expected},
		Vars:  []*autofunc.Variable{actual},
		Input: actual,
	}
	funcTest.FullCheck(t)
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
	funcTest := &functest.RFuncChecker{
		F:     meanSquaredTestFunc{expected},
		Vars:  []*autofunc.Variable{actual},
		Input: actual,
		RV:    rVector,
	}
	funcTest.FullCheck(t)
}

func TestTotalCostBatcher(t *testing.T) {
	net := Network{NewDenseLayer(2, 3)}
	samples := sgd.SliceSampleSet{
		VectorSample{Input: []float64{1, -1}, Output: []float64{1, -2, 0.4}},
		VectorSample{Input: []float64{1, 1}, Output: []float64{1, -3, -0.4}},
		VectorSample{Input: []float64{-1, -1}, Output: []float64{0, -2, 0.4}},
		VectorSample{Input: []float64{-1, 1}, Output: []float64{1, -2, 0.9}},
		VectorSample{Input: []float64{0.5, 0.75}, Output: []float64{-1, 2, 0.4}},
	}
	cf := MeanSquaredCost{}
	expected := TotalCost(cf, net, samples)
	for _, batchSize := range []int{1, 0, 3, 5, 10} {
		actual := TotalCostBatcher(cf, net.BatchLearner(), samples, batchSize)
		if math.Abs(actual-expected) > 1e-5 {
			t.Errorf("batch %d: expected %v got %v", batchSize, expected, actual)
		}
	}
}
