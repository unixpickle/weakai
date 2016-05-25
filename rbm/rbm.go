package rbm

import (
	"math"
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
)

// An RBM stores the parameters of a
// Restricted Boltzmann Machine.
type RBM struct {
	Weights       *linalg.Matrix
	HiddenBiases  linalg.Vector
	VisibleBiases linalg.Vector
}

func NewRBM(visibleCount, hiddenCount int) *RBM {
	return &RBM{
		Weights:       linalg.NewMatrix(hiddenCount, visibleCount),
		HiddenBiases:  make(linalg.Vector, hiddenCount),
		VisibleBiases: make(linalg.Vector, visibleCount),
	}
}

// SampleVisible generates a random visible vector
// given a vector of hidden layer values.
// The visible vector will be written to output,
// allowing the caller to cache a slice for visible
// samples.
func (r *RBM) SampleVisible(output, hiddenValues []bool) {
	expected := r.ExpectedVisible(hiddenValues)
	sampleVector(output, expected)
}

// SampleHidden generates a random hidden vector
// given a vector of visible values.
// The hidden values will be written to output,
// allowing the caller to cache a slice for hidden
// samples.
func (r *RBM) SampleHidden(output, visibleValues []bool) {
	expected := r.ExpectedHidden(visibleValues)
	sampleVector(output, expected)
}

// ExpectedVisible returns the expected value of
// the visible layer given a hidden vector.
func (r *RBM) ExpectedVisible(hidden []bool) linalg.Vector {
	hiddenRow := &linalg.Matrix{
		Rows: 1,
		Cols: len(hidden),
		Data: make([]float64, len(hidden)),
	}
	for i, b := range hidden {
		if b {
			hiddenRow.Data[i] = 1
		}
	}
	matrixProduct := hiddenRow.Mul(r.Weights)
	weighted := linalg.Vector(matrixProduct.Data)
	expected := weighted.Add(r.VisibleBiases)

	mapSigmoid(expected)

	return expected
}

// ExpectedHidden returns the expected value of
// the hidden layer given a visible vector.
func (r *RBM) ExpectedHidden(visible []bool) linalg.Vector {
	visibleCol := &linalg.Matrix{
		Rows: len(visible),
		Cols: 1,
		Data: make([]float64, len(visible)),
	}
	for i, b := range visible {
		if b {
			visibleCol.Data[i] = 1
		}
	}
	matrixProduct := r.Weights.Mul(visibleCol)
	weighted := linalg.Vector(matrixProduct.Data)
	expected := weighted.Add(r.HiddenBiases)

	mapSigmoid(expected)

	return expected
}

func mapSigmoid(v linalg.Vector) {
	for i, x := range v {
		e := math.Exp(x)
		v[i] = e / (1 + e)
	}
}

func sampleVector(output []bool, expected linalg.Vector) {
	for i, prob := range expected {
		if rand.Float64() >= prob {
			output[i] = false
		} else {
			output[i] = true
		}
	}
}
