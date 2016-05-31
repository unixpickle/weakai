package rnn

import (
	"errors"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

// A DeepRNN is multiple RNNs stacked on top
// of each other.
// The first RNN is the "visible" layer, which
// receives inputs, while the last RNN is the
// "output" layer, which gives outputs.
type DeepRNN []RNN

func DeserializeDeepRNN(d []byte) (serializer.Serializer, error) {
	list, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	res := make(DeepRNN, len(list))
	for i, x := range list {
		if r, ok := x.(RNN); ok {
			res[i] = r
		} else {
			return nil, errors.New("slice contained an object which is not an RNN")
		}
	}
	return res, nil
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

func (d DeepRNN) Serialize() ([]byte, error) {
	s := make([]serializer.Serializer, len(d))
	for i, x := range d {
		s[i] = x
	}
	return serializer.SerializeSlice(s)
}

func (d DeepRNN) SerializerType() string {
	return serializerTypeDeepRNN
}
