package neuralnet

import (
	"math"

	"github.com/unixpickle/autofunc"
)

const (
	adamDefaultDecayRate1 = 0.9
	adamDefaultDecayRate2 = 0.999
	adamDefaultDamping    = 1e-8
)

// Adam implements the adaptive moments SGD technique
// described in https://arxiv.org/pdf/1412.6980.pdf.
type Adam struct {
	Gradienter Gradienter

	// These are decay rates for the first and second
	// moments of the gradient.
	// If these are 0, defaults as suggested in the
	// original Adam paper are used.
	DecayRate1, DecayRate2 float64

	// Damping is used to prevent divisions by zero.
	// This should be very small.
	// If it is 0, a default is used.
	Damping float64

	firstMoment  autofunc.Gradient
	secondMoment autofunc.Gradient
	iteration    float64
}

func (a *Adam) Gradient(s SampleSet) autofunc.Gradient {
	realGradient := a.Gradienter.Gradient(s)
	a.updateMoments(realGradient)

	a.iteration++
	scalingFactor := math.Sqrt(1-math.Pow(a.decayRate(2), a.iteration)) /
		(1 - math.Pow(a.decayRate(1), a.iteration))
	damping := a.damping()
	for variable, vec := range realGradient {
		firstVec := a.firstMoment[variable]
		secondVec := a.secondMoment[variable]
		for i, x := range firstVec {
			vec[i] = scalingFactor * x / math.Sqrt(secondVec[i]+damping)
		}
	}

	return realGradient
}

func (a *Adam) updateMoments(grad autofunc.Gradient) {
	if a.firstMoment == nil {
		a.firstMoment = grad.Copy()
		a.firstMoment.Scale(1 - a.decayRate(1))
	} else {
		decayRate := a.decayRate(1)
		a.firstMoment.Scale(decayRate)

		keepRate := 1 - decayRate
		for variable, vec := range grad {
			momentVec := a.firstMoment[variable]
			for i, x := range vec {
				momentVec[i] += keepRate * x
			}
		}
	}
	if a.secondMoment == nil {
		a.secondMoment = grad.Copy()
		for _, v := range a.secondMoment {
			for i, x := range v {
				v[i] = x * x
			}
		}
		a.secondMoment.Scale(1 - a.decayRate(2))
	} else {
		decayRate := a.decayRate(2)
		a.secondMoment.Scale(decayRate)

		keepRate := 1 - decayRate
		for variable, vec := range grad {
			momentVec := a.secondMoment[variable]
			for i, x := range vec {
				momentVec[i] += keepRate * x * x
			}
		}
	}
}

func (a *Adam) decayRate(moment int) float64 {
	if moment == 1 {
		if a.DecayRate1 == 0 {
			return adamDefaultDecayRate1
		} else {
			return a.DecayRate1
		}
	} else if moment == 2 {
		if a.DecayRate2 == 0 {
			return adamDefaultDecayRate2
		} else {
			return a.DecayRate2
		}
	} else {
		panic("invalid moment.")
	}
}

func (a *Adam) damping() float64 {
	if a.Damping != 0 {
		return a.Damping
	} else {
		return adamDefaultDamping
	}
}
