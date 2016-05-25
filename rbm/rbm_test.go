package rbm

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/num-analysis/kahan"
	"github.com/unixpickle/num-analysis/linalg"
)

const (
	rbmTestVisibleSize = 3
	rbmTestHiddenSize  = 4
)

func TestRBMExpectedVisible(t *testing.T) {
	r := NewRBM(rbmTestVisibleSize, rbmTestHiddenSize)
	r.Randomize(1)
	for i := range r.HiddenBiases {
		r.HiddenBiases[i] = rand.Float64()
	}
	for i := range r.VisibleBiases {
		r.VisibleBiases[i] = rand.Float64()
	}
	for i := 0; i < (1 << rbmTestHiddenSize); i++ {
		hidden := boolVecFromInt(i, rbmTestHiddenSize)
		actual := r.ExpectedVisible(hidden)
		expected := rbmExactExpectedVisible(r, hidden)
		for i, x := range expected {
			a := actual[i]
			if math.IsNaN(a) || math.Abs(a-x) > 1e-5 {
				t.Fatalf("invalid expectation for hidden layer (expected %f got %f)", x, a)
			}
		}
	}
}

func TestRBMExpectedHidden(t *testing.T) {
	r := NewRBM(rbmTestVisibleSize, rbmTestHiddenSize)
	r.Randomize(1)
	for i := range r.HiddenBiases {
		r.HiddenBiases[i] = rand.Float64()
	}
	for i := range r.VisibleBiases {
		r.VisibleBiases[i] = rand.Float64()
	}
	for i := 0; i < (1 << rbmTestVisibleSize); i++ {
		visible := boolVecFromInt(i, rbmTestVisibleSize)
		actual := r.ExpectedHidden(visible)
		expected := rbmExactExpectedHidden(r, visible)
		for i, x := range expected {
			a := actual[i]
			if math.IsNaN(a) || math.Abs(a-x) > 1e-5 {
				t.Fatalf("invalid expectation for hidden layer (expected %f got %f)", x, a)
			}
		}
	}
}

func rbmExactExpectedHidden(r *RBM, visible []bool) linalg.Vector {
	return rbmExactExpectation(r, visible, true)
}

func rbmExactExpectedVisible(r *RBM, hidden []bool) linalg.Vector {
	return rbmExactExpectation(r, hidden, false)
}

func rbmExactExpectation(r *RBM, layer []bool, hidden bool) linalg.Vector {
	var normalizer kahan.Summer64
	var outcomeSum []kahan.Summer64
	if hidden {
		outcomeSum = make([]kahan.Summer64, rbmTestHiddenSize)
	} else {
		outcomeSum = make([]kahan.Summer64, rbmTestVisibleSize)
	}

	for i := 0; i < (1 << uint(len(outcomeSum))); i++ {
		variableVec := boolVecFromInt(i, len(outcomeSum))
		var prob float64
		if hidden {
			prob = math.Exp(-rbmEnergy(r, layer, variableVec))
		} else {
			prob = math.Exp(-rbmEnergy(r, variableVec, layer))
		}
		normalizer.Add(prob)
		for j, b := range variableVec {
			if b {
				outcomeSum[j].Add(prob)
			}
		}
	}

	expectation := make(linalg.Vector, len(outcomeSum))
	norm := 1.0 / normalizer.Sum()
	for i, s := range outcomeSum {
		expectation[i] = norm * s.Sum()
	}
	return expectation
}

func rbmEnergy(r *RBM, input, output []bool) float64 {
	inputVec := make(linalg.Vector, len(input))
	for i, x := range input {
		if x {
			inputVec[i] = 1
		}
	}
	outputVec := make(linalg.Vector, len(output))
	for i, x := range output {
		if x {
			outputVec[i] = 1
		}
	}

	energy := inputVec.Dot(r.VisibleBiases)
	energy += outputVec.Dot(r.HiddenBiases)

	inputCol := linalg.NewMatrixColumn(inputVec)
	energy += outputVec.Dot(linalg.Vector(r.Weights.Mul(inputCol).Data))

	return -energy
}

func boolVecFromInt(fields, size int) []bool {
	res := make([]bool, size)
	for i := 0; i < size; i++ {
		if (fields & (1 << uint(i))) != 0 {
			res[i] = true
		}
	}
	return res
}
