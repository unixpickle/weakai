package neuralnet

import (
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
)

// Network is a Layer formed by composing many
// other layers.
// It also facilitates aggregate operations on
// the underlying layers through its Randomize()
// and Parameters() methods.
type Network []Layer

func DeserializeNetwork(data []byte) (Network, error) {
	var res Network

	slice, err := serializer.DeserializeSlice(data)
	if err != nil {
		return nil, err
	}

	for _, x := range slice {
		if layer, ok := x.(Layer); ok {
			res = append(res, layer)
		} else {
			return nil, errors.New("slice element is not a Layer")
		}
	}

	return res, nil
}

// Randomize calls Randomize() on all the layers
// in n that implement Randomizer.
func (n Network) Randomize() {
	for _, layer := range n {
		if r, ok := layer.(Randomizer); ok {
			r.Randomize()
		}
	}
}

// Parameters concatenates the parameters of
// every Learner in n.
func (n Network) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, layer := range n {
		if l, ok := layer.(Learner); ok {
			res = append(res, l.Parameters()...)
		}
	}
	return res
}

func (n Network) Apply(in autofunc.Result) autofunc.Result {
	for _, layer := range n {
		in = layer.Apply(in)
	}
	return in
}

func (n Network) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	for _, layer := range n {
		in = layer.ApplyR(v, in)
	}
	return in
}

func (n Network) SetCache(c *autofunc.VectorCache) {
	for _, layer := range n {
		layer.SetCache(c)
	}
}

func (n Network) Serialize() ([]byte, error) {
	serializers := make([]serializer.Serializer, len(n))
	for i, x := range n {
		serializers[i] = x
	}
	return serializer.SerializeSlice(serializers)
}

func (n Network) SerializerType() string {
	return serializerTypeNetwork
}
