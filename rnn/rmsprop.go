package rnn

import (
	"math"

	"github.com/unixpickle/num-analysis/linalg"
)

// DefaultRMSPropResiliency is the default
// Resiliency attribute for RMSProp.
const DefaultRMSPropResiliency = 0.9

// RMSProp is a mini-batch training algorithm
// which attempts to improve on SGD by dividing
// partial derivatives by their magnitudes.
// This is the mini-batch version of using the
// signs of the partials and discarding the
// magnitudes altogether.
type RMSProp struct {
	// SGD stores the parameters for SGD,
	// since RMSProp is basically a normalized
	// version of SGD.
	SGD

	// Resiliency is used when averaging the
	// squares of gradient components.
	// A resiliency very close to 1.0 prevents
	// the mean square partials from changing
	// very much, while a resiliency closer to
	// 0.0 throws away old mean square partials
	// fairly quickly.
	//
	// If this is 0, DefaultRMSPropResiliency is used.
	Resiliency float64

	// ParamMean is the current mean of the squares
	// of the parameter partials.
	// This should start as nil at the beginning
	// of training.
	Mean []linalg.Vector
}

// Train calls Train on the underlying SGD,
// setting the SGD's normalizer to the
// RMSProp normalization routine.
func (r *RMSProp) Train(n RNN) {
	r.SGD.Normalizer = r.normalize
	r.SGD.Train(n)
}

// TrainSynchronously is like Train, but uses
// the SGD's TrainSynchronously() method.
func (r *RMSProp) TrainSynchronously(n RNN) {
	r.SGD.Normalizer = r.normalize
	r.SGD.TrainSynchronously(n)
}

func (r *RMSProp) normalize(g Gradient) {
	if r.Mean == nil {
		params := g.Params()
		r.Mean = make([]linalg.Vector, len(params))
		for i, p := range params {
			r.Mean[i] = make(linalg.Vector, len(p))
			for j, x := range p {
				r.Mean[i][j] = x * x
			}
		}
	} else {
		resil := r.Resiliency
		if resil == 0 {
			resil = DefaultRMSPropResiliency
		}
		params := g.Params()
		for i, p := range params {
			for j, x := range p {
				r.Mean[i][j] = r.Mean[i][j]*resil + (1-resil)*x*x
			}
		}
	}
	for paramIdx, v := range g.Params() {
		for i, x := range v {
			v[i] = x / math.Sqrt(r.Mean[paramIdx][i])
		}
	}
}
