package lstm

import "github.com/unixpickle/num-analysis/linalg"

// DeepGradient is a collection of gradients
// where each gradient corresponds to another
// RNN in a DeepRNN.
type DeepGradient []*Gradient

func (d DeepGradient) Scale(f float64) {
	for _, r := range d {
		r.Scale(f)
	}
}

func (d DeepGradient) Add(d1 DeepGradient) {
	for i, r := range d {
		r.Add(d1[i])
	}
}

// A DeepRNN is multiple RNNs stacked on top
// of each other.
// The first RNN is the "visible" layer, which
// receives inputs, while the last RNN is the
// "output" layer, which gives outputs.
type DeepRNN []*RNN

func NewDeepRNN(inputSize, outputSize int, hiddenSizes ...int) DeepRNN {
	var res DeepRNN
	for i, h := range hiddenSizes {
		inSize := inputSize
		if i > 0 {
			inSize = hiddenSizes[i-1]
		}
		outSize := outputSize
		if i+1 < len(hiddenSizes) {
			outSize = hiddenSizes[i+1]
		}
		res = append(res, NewRNN(inSize, h, outSize))
	}
	return res
}

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

func (d DeepRNN) CostGradient(costPartials []linalg.Vector) DeepGradient {
	var res DeepGradient
	for i := len(d) - 1; i >= 0; i-- {
		r := d[i]
		grad := r.CostGradient(costPartials)
		res = append(res, grad)
		costPartials = grad.Inputs
	}
	return res
}

func (d DeepRNN) StepGradient(g DeepGradient) {
	for i, r := range d {
		r.StepGradient(g[i])
	}
}

func (d DeepRNN) Reset() {
	for _, r := range d {
		r.Reset()
	}
}
