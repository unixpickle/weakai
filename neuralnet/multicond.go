package neuralnet

import (
	"math"

	"github.com/unixpickle/autofunc"
)

type Multicond struct {
	RGradienter RGradienter
	Learner     Learner

	Intervals int
	Damping   float64
}

func (m *Multicond) Gradient(s SampleSet) autofunc.Gradient {
	grad := m.RGradienter.Gradient(s)
	var rg autofunc.RGradient
	for i := 0; i < m.Intervals; i++ {
		grad, rg = m.RGradienter.RGradient(autofunc.RVector(grad.Copy()), s)
		for variable, vec := range rg {
			gv := grad[variable]
			for i, x := range vec {
				gv[i] /= math.Abs(x)*(1-m.Damping) + m.Damping
			}
		}
	}
	return grad
}
