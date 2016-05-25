package rbm

import (
	"math"
	"math/rand"

	"github.com/unixpickle/num-analysis/kahan"
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

// Randomize initializes the weights randomly.
// The random values will be clamped to
// the range [-randMag, randMag].
func (r *RBM) Randomize(randMag float64) {
	for i := range r.Weights.Data {
		r.Weights.Data[i] = rand.Float64()*randMag*2 - randMag
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
	result := make(linalg.Vector, len(r.VisibleBiases))
	for i := range result {
		var sum kahan.Summer64
		for j, h := range hidden {
			if h {
				sum.Add(r.Weights.Get(j, i))
			}
		}
		result[i] = sum.Sum()
	}

	result.Add(r.VisibleBiases)
	mapSigmoid(result)

	return result
}

// ExpectedHidden returns the expected value of
// the hidden layer given a visible vector.
func (r *RBM) ExpectedHidden(visible []bool) linalg.Vector {
	result := make(linalg.Vector, len(r.HiddenBiases))
	for i := range result {
		var sum kahan.Summer64
		for j, v := range visible {
			if v {
				sum.Add(r.Weights.Get(i, j))
			}
		}
		result[i] = sum.Sum()
	}

	result.Add(r.HiddenBiases)
	mapSigmoid(result)

	return result
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
