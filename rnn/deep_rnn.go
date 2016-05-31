package rnn

import "github.com/unixpickle/num-analysis/linalg"

// A DeepRNN is multiple RNNs stacked on top
// of each other.
// The first RNN is the "visible" layer, which
// receives inputs, while the last RNN is the
// "output" layer, which gives outputs.
type DeepRNN []RNN

func (d DeepRNN) Randomize() {
	for _, r := range d {
		r.Randomize()
	}
}

func (d DeepRNN) StepTime(in linalg.Vector) linalg.Vector {
	for _, r := range d {
		in = r.StepTime(in)
	}
	return in
}

func (d DeepRNN) CostGradient(costPartials []linalg.Vector) Gradient {
	var res DeepGradient
	for i := len(d) - 1; i >= 0; i-- {
		r := d[i]
		grad := r.CostGradient(costPartials)
		res = append(res, grad)
		costPartials = grad.Inputs()
	}
	return res
}

func (d DeepRNN) Reset() {
	for _, r := range d {
		r.Reset()
	}
}

func (d DeepRNN) StepGradient(g DeepGradient) {
	for i, r := range d {
		r.StepGradient(g[len(d)-(i+1)])
	}
}
