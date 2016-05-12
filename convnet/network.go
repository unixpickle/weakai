package convnet

import (
	"fmt"

	"github.com/unixpickle/num-analysis/kahan"
)

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
	n := &Network{Layers: make([]Layer, len(prototypes))}
	for i, proto := range prototypes {
		layer := proto.Make()
		if i != 0 {
			ok := layer.SetInput(n.Layers[i-1].Output())
			if !ok {
				return nil, fmt.Errorf("layer %d cannot feed into layer %d", i-1, i)
			}

			ok = n.Layers[i-1].SetDownstreamGradient(layer.UpstreamGradient())
			if !ok {
				return nil, fmt.Errorf("layer %d cannot feed back into layer %d", i, i-1)
			}
		}
	}
	return n, nil
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

func (n *Network) PropagateBackward() {
	for i := len(n.Layers) - 1; i >= 0; i-- {
		n.Layers[i].PropagateBackward()
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
