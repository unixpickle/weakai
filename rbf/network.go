package rbf

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var n Network
	serializer.RegisterTypedDeserializer(n.SerializerType(), DeserializeNetwork)
}

// A Network is an RBF network, with a radial basis layer
// represented by a Dist-Scale-Exp layer combo, and the
// classification layer as a *neuralnet.DenseLayer.
type Network struct {
	DistLayer  *DistLayer
	ScaleLayer *ScaleLayer
	ExpLayer   *ExpLayer
	OutLayer   *neuralnet.DenseLayer
}

// DeserializeNetwork deserializes a Network.
func DeserializeNetwork(d []byte) (*Network, error) {
	var n Network
	err := serializer.DeserializeAny(d, &n.DistLayer, &n.ScaleLayer, &n.ExpLayer, &n.OutLayer)
	if err != nil {
		return nil, err
	}
	return &n, nil
}

// Apply applies the network to an input.
func (n *Network) Apply(in autofunc.Result) autofunc.Result {
	comp := autofunc.ComposedFunc{n.DistLayer, n.ScaleLayer, n.ExpLayer, n.OutLayer}
	return comp.Apply(in)
}

// ApplyR applies the network to an input.
func (n *Network) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	comp := autofunc.ComposedRFunc{n.DistLayer, n.ScaleLayer, n.ExpLayer, n.OutLayer}
	return comp.ApplyR(rv, in)
}

// Batch applies the network in batch.
func (n *Network) Batch(in autofunc.Result, m int) autofunc.Result {
	comp := autofunc.ComposedBatcher{
		n.DistLayer,
		&autofunc.FuncBatcher{
			F: autofunc.ComposedFunc{n.ScaleLayer, n.ExpLayer},
		},
		n.OutLayer,
	}
	return comp.Batch(in, m)
}

// BatchR applies the network in batch.
func (n *Network) BatchR(rv autofunc.RVector, in autofunc.RResult, m int) autofunc.RResult {
	comp := autofunc.ComposedRBatcher{
		n.DistLayer,
		&autofunc.RFuncBatcher{
			F: autofunc.ComposedRFunc{n.ScaleLayer, n.ExpLayer},
		},
		n.OutLayer,
	}
	return comp.BatchR(rv, in, m)
}

// Parameters returns all of the network's parameters.
func (n *Network) Parameters() []*autofunc.Variable {
	learners := []sgd.Learner{n.DistLayer, n.ScaleLayer, n.OutLayer}
	var res []*autofunc.Variable
	for _, l := range learners {
		res = append(res, l.Parameters()...)
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a Network with the serializer package.
func (n *Network) SerializerType() string {
	return "github.com/unixpickle/weakai/rbf.Network"
}

// Serialize serializes the network.
func (n *Network) Serialize() ([]byte, error) {
	return serializer.SerializeAny(n.DistLayer, n.ScaleLayer, n.ExpLayer, n.OutLayer)
}
