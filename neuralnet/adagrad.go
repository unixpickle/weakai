package neuralnet

import (
	"math"

	"github.com/unixpickle/autofunc"
)

// AdaGrad is a Gradienter which implements the
// AdaGrad SGD algorithm.
type AdaGrad struct {
	Gradienter Gradienter
	Damping    float64

	squaredHistory autofunc.Gradient
}

func (a *AdaGrad) Gradient(s SampleSet) autofunc.Gradient {
	actualGrad := a.Gradienter.Gradient(s)

	if a.squaredHistory == nil {
		a.squaredHistory = actualGrad.Copy()
		for _, v := range a.squaredHistory {
			for i, x := range v {
				v[i] *= x
			}
		}
	} else {
		for variable, vec := range actualGrad {
			histVec := a.squaredHistory[variable]
			for i, x := range vec {
				histVec[i] += x * x
			}
		}
	}

	for variable, vec := range actualGrad {
		histVec := a.squaredHistory[variable]
		for i, x := range histVec {
			vec[i] /= math.Sqrt(x) + a.Damping
		}
	}

	return actualGrad
}
