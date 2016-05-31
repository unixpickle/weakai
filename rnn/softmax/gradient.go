package softmax

import (
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

type Gradient []linalg.Vector

func (g Gradient) Inputs() []linalg.Vector {
	return g
}

func (g Gradient) Scale(f float64) rnn.Gradient {
	for _, x := range g {
		x.Scale(f)
	}
	return g
}

func (g Gradient) Add(g1Interface rnn.Gradient) rnn.Gradient {
	g1 := g1Interface.(Gradient)
	for i, x := range g {
		x.Add(g1[i])
	}
	return g
}

func (g Gradient) LargestComponent() float64 {
	return 0
}

func (g Gradient) ClipComponents(m float64) {
}
