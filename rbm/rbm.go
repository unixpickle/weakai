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

// SampleVisible generates a random visible vector
// given a vector of hidden layer values.
// The visible vector will be written to output,
// allowing the caller to cache a slice for visible
// samples.
func (r *RBM) SampleVisible(output, hiddenValues []bool) {
	r.sample(output, hiddenValues, r.VisibleBiases)
}

// SampleHidden generates a random hidden vector
// given a vector of visible values.
// The hidden values will be written to output,
// allowing the caller to cache a slice for hidden
// samples.
func (r *RBM) SampleHidden(output, visibleValues []bool) {
	r.sample(output, visibleValues, r.HiddenBiases)
}

func (r *RBM) sample(output, input []bool, biases linalg.Vector) {
	inputCol := &linalg.Matrix{
		Rows: len(input),
		Cols: 1,
		Data: make([]float64, len(input)),
	}
	for i, b := range input {
		if b {
			inputCol.Data[i] = 1
		}
	}
	matrixProduct := r.Weights.Mul(inputCol)
	weightedInput := linalg.Vector(matrixProduct.Data)
	energyVec := weightedInput.Add(biases)

	for i, energy := range energyVec {
		energyExp := math.Exp(energy)
		prob := energyExp / (1 + energyExp)
		if rand.Float64() >= prob {
			output[i] = false
		} else {
			output[i] = true
		}
	}
}
