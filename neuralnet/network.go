package neuralnet

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/unixpickle/num-analysis/kahan"
	"github.com/unixpickle/serializer"
)

var networkEncodingEndian = binary.LittleEndian

// A Network is a feed-forward neural network
// composed of a series of layers, each one
// feeding into the next one.
//
// A Network itself acts as a Layer, with
// its input feeding into the first layer
// and its output coming out of the last one.
type Network struct {
	Layers []Layer
}

// NewNetwork generates a network by creating
// and connecting a bunch of layers using their
// respective prototypes.
//
// If layer dimensions do not match, this returns
// an error describing the issue.
func NewNetwork(prototypes []LayerPrototype) (*Network, error) {
	n := &Network{Layers: make([]Layer, 0, len(prototypes))}
	for _, proto := range prototypes {
		layer := proto.Make()
		if err := n.AddLayer(layer); err != nil {
			return nil, err
		}
	}
	return n, nil
}

func DeserializeNetwork(data []byte) (*Network, error) {
	res := &Network{}

	slice, err := serializer.DeserializeSlice(data)
	if err != nil {
		return nil, err
	}

	for _, x := range slice {
		if layer, ok := x.(Layer); ok {
			if err := res.AddLayer(layer); err != nil {
				return nil, err
			}
		} else {
			return nil, errors.New("slice element is not Layer")
		}
	}

	return res, nil
}

// AddLayer adds a layer to the tail of this network.
// This fails if the layer's dimensionality does not
// match the previous layer's output dimensions.
func (n *Network) AddLayer(l Layer) error {
	idx := len(n.Layers)
	n.Layers = append(n.Layers, l)
	if idx != 0 {
		ok := n.Layers[idx].SetInput(n.Layers[idx-1].Output())
		if !ok {
			return fmt.Errorf("layer %d cannot feed into layer %d", idx-1, idx)
		}
		ok = n.Layers[idx-1].SetDownstreamGradient(n.Layers[idx].UpstreamGradient())
		if !ok {
			return fmt.Errorf("layer %d cannot feed back into layer %d", idx, idx-1)
		}
	}
	return nil
}

func (n *Network) Randomize() {
	for _, layer := range n.Layers {
		layer.Randomize()
	}
}

func (n *Network) PropagateForward() {
	for _, l := range n.Layers {
		l.PropagateForward()
	}
}

func (n *Network) PropagateBackward(upstream bool) {
	for i := len(n.Layers) - 1; i >= 0; i-- {
		if i != 0 || upstream {
			n.Layers[i].PropagateBackward(true)
		} else {
			n.Layers[i].PropagateBackward(false)
		}
	}
}

func (n *Network) Output() []float64 {
	return n.Layers[len(n.Layers)-1].Output()
}

func (n *Network) UpstreamGradient() []float64 {
	return n.Layers[0].UpstreamGradient()
}

func (n *Network) Input() []float64 {
	return n.Layers[0].Input()
}

func (n *Network) SetInput(v []float64) bool {
	return n.Layers[0].SetInput(v)
}

func (n *Network) DownstreamGradient() []float64 {
	return n.Layers[len(n.Layers)-1].DownstreamGradient()
}

func (n *Network) SetDownstreamGradient(v []float64) bool {
	return n.Layers[len(n.Layers)-1].SetDownstreamGradient(v)
}

func (n *Network) GradientMagSquared() float64 {
	sum := kahan.NewSummer64()
	for _, l := range n.Layers {
		sum.Add(l.GradientMagSquared())
	}
	return sum.Sum()
}

func (n *Network) StepGradient(f float64) {
	for _, l := range n.Layers {
		l.StepGradient(f)
	}
}

func (n *Network) Alias() Layer {
	net := &Network{}
	for _, l := range n.Layers {
		net.AddLayer(l.Alias())
	}
	return net
}

func (n *Network) Serialize() ([]byte, error) {
	serializers := make([]serializer.Serializer, len(n.Layers))
	for i, x := range n.Layers {
		serializers[i] = x
	}
	return serializer.SerializeSlice(serializers)
}

func (n *Network) SerializerType() string {
	return serializerTypeNetwork
}
