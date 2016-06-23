package neuralnet

import (
	"math"
	"math/rand"

	"github.com/unixpickle/autofunc"
)

// Equilibration pre-conditions gradient descent by
// attempting to equilibrate the rows of the Hessian.
//
// This algorithm is described in this paper:
// http://arxiv.org/pdf/1502.04390v2.pdf
type Equilibration struct {
	RGradienter RGradienter
	Learner     Learner

	// Accumulate indicates whether to use AdaGrad-
	// style sums or RMSProp style exponential decay.
	// If Accumulate is true, then the normalization
	// coefficients are continually added together,
	// resulting in a naturally decaying step size.
	// If it is false, then Memory is used to decay
	// old normalization coefficients and use new ones.
	Accumulate bool

	// If Accumulate is set to false, then Memory is a
	// ratio (between 0 and 1) of how much new normalizing
	// coefficients should override old ones.
	//
	// A value of 0 means that, at each update interval,
	// a completely new approximation of the magnitudes
	// of the Hessian's rows will be taken.
	Memory float64

	// UpdateInterval is the number of Gradient calls
	// between each time the noramlizing coefficients
	// are re-calculated.
	// A value of 0 means that the coefficients are
	// updated at each iteration.
	UpdateInterval int

	// NumSamples specifies the number of times the
	// Hessian's rows' magnitudes will be sampled at
	// each update interval.
	// If this is 0, a default value of 1 is used.
	NumSamples int

	// Damping specifies how much the equilibrated
	// coefficients should be ignored.
	// The higher the damping value, the less effect
	// equilibration will have.
	// A reasonable value is something relatively
	// small, like 1e-4.
	// A damping factor of 0 may work for some things,
	// but it can cause numerical problems when a
	// parameter has a very small row in the Hessian.
	Damping float64

	lastUpdate int
	rCache     autofunc.RVector
	squareMags autofunc.RGradient
}

func (e *Equilibration) Gradient(s SampleSet) autofunc.Gradient {
	var rawGrad autofunc.Gradient
	if e.squareMags == nil || e.lastUpdate == e.UpdateInterval {
		rawGrad = e.updateSquareMags(s)
		e.lastUpdate = 0
	} else {
		e.lastUpdate++
		rawGrad = e.RGradienter.Gradient(s)
	}

	for variable, vector := range rawGrad {
		rMags := e.squareMags[variable]
		for i, x := range rMags {
			if x != 0 {
				coeff := math.Sqrt(x)
				vector[i] /= coeff*(1-e.Damping) + e.Damping
			}
		}
	}

	return rawGrad
}

func (e *Equilibration) updateSquareMags(s SampleSet) autofunc.Gradient {
	if e.rCache == nil {
		params := e.Learner.Parameters()
		e.rCache = autofunc.RVector(autofunc.NewGradient(params))
	}
	e.randomizeRVector()

	sampleCount := e.NumSamples
	if sampleCount == 0 {
		sampleCount = 1
	}

	var grad autofunc.Gradient
	var rGrad autofunc.RGradient
	for i := 0; i < sampleCount; i++ {
		g, rg := e.RGradienter.RGradient(e.rCache, s)
		grad = g
		squareRGrad(rg)
		if sampleCount == 1 {
			rGrad = rg
		} else {
			if i == 0 {
				rGrad = rg.Copy()
			} else {
				rGrad.Add(rg)
			}
		}
	}

	if sampleCount > 1 {
		rGrad.Scale(1 / float64(sampleCount))
	}

	if e.squareMags == nil {
		if sampleCount > 1 {
			e.squareMags = rGrad
		} else {
			e.squareMags = rGrad.Copy()
		}
	} else {
		if !e.Accumulate {
			e.squareMags.Scale(e.Memory)
			rGrad.Scale(1 - e.Memory)
		}
		e.squareMags.Add(rGrad)
	}

	return grad
}

func (e *Equilibration) randomizeRVector() {
	for _, vec := range e.rCache {
		for i := range vec {
			vec[i] = rand.NormFloat64()
		}
	}
}

func squareRGrad(rg autofunc.RGradient) {
	for _, v := range rg {
		for i, x := range v {
			v[i] = x * x
		}
	}
}
