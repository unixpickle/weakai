package rnn

import "github.com/unixpickle/num-analysis/linalg"

// DeepGradient is a collection of gradients
// where each gradient corresponds to another
// RNN in a DeepRNN.
// The first Gradient in a DeepGradient is from
// the last RNN in a DeepRNN.
type DeepGradient []Gradient

func (d DeepGradient) Inputs() []linalg.Vector {
	return d[len(d)-1].Inputs()
}

func (d DeepGradient) Params() []linalg.Vector {
	var res []linalg.Vector
	for _, g := range d {
		res = append(res, g.Params()...)
	}
	return res
}
