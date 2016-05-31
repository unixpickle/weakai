package rnn

import (
	"math"

	"github.com/unixpickle/num-analysis/linalg"
)

// DeepGradient is a collection of gradients
// where each gradient corresponds to another
// RNN in a DeepRNN.
// The first Gradient in a DeepGradient is from
// the last RNN in a DeepRNN.
type DeepGradient []Gradient

func (d DeepGradient) Inputs() []linalg.Vector {
	return d[len(d)-1].Inputs()
}

func (d DeepGradient) Scale(f float64) Gradient {
	for _, r := range d {
		r.Scale(f)
	}
	return d
}

func (d DeepGradient) Add(d1Interface Gradient) Gradient {
	d1 := d1Interface.(DeepGradient)
	for i, r := range d {
		r.Add(d1[i])
	}
	return d
}

func (d DeepGradient) LargestComponent() float64 {
	var largest float64
	for _, r := range d {
		largest = math.Max(r.LargestComponent(), largest)
	}
	return largest
}

func (d DeepGradient) ClipComponents(m float64) {
	for _, r := range d {
		r.ClipComponents(m)
	}
}
