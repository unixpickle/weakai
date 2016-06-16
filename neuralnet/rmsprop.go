package neuralnet

import (
	"math"

	"github.com/unixpickle/autofunc"
)

const defaultRMSPropResiliency = 0.9

// RMSProp is a Gradienter which wraps another
// Gradienter and attempts to pre-condition SGD
// by dividing weights by a rolling average of
// their previous and current values.
type RMSProp struct {
	Gradienter Gradienter

	// Resiliency is used when averaging the
	// squares of gradient components.
	// A resiliency very close to 1.0 prevents
	// the mean square partials from changing
	// very much, while a resiliency closer to
	// 0.0 throws away old mean square partials
	// fairly quickly.
	//
	// If this is 0, a default is used.
	Resiliency float64

	// RollingAverage is the current average of
	// the squares of the gradient entries.
	// You probably want to leave this as nil
	// so that it gets set automatically after
	// the first batch.
	RollingAverage autofunc.Gradient
}

func (r *RMSProp) Gradient(s SampleSet) autofunc.Gradient {
	grad := r.Gradienter.Gradient(s)
	squaredGrad := grad.Copy()
	for _, v := range squaredGrad {
		for i, x := range v {
			v[i] *= x
		}
	}

	if r.RollingAverage != nil {
		resil := r.Resiliency
		if resil == 0 {
			resil = defaultRMSPropResiliency
		}
		r.RollingAverage.Scale(resil)
		squaredGrad.Scale(1 - resil)
		r.RollingAverage.Add(squaredGrad)
	} else {
		r.RollingAverage = squaredGrad
	}

	for variable, vec := range r.RollingAverage {
		gradVec := grad[variable]
		for i, x := range vec {
			if x != 0 {
				gradVec[i] /= math.Sqrt(x)
			}
		}
	}

	return grad
}
