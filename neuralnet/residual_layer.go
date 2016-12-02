package neuralnet

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
)

// A ResidualLayer wraps a set of layers and adds its input
// the layers' final output.
type ResidualLayer struct {
	Network Network
}

// DeserializeResidualLayer deserializes a ResidualLayer.
func DeserializeResidualLayer(d []byte) (*ResidualLayer, error) {
	var n Network
	if err := serializer.DeserializeAny(d, &n); err != nil {
		return nil, err
	}
	return &ResidualLayer{Network: n}, nil
}

// Apply applies the layer.
func (r *ResidualLayer) Apply(in autofunc.Result) autofunc.Result {
	return autofunc.Pool(in, func(inPool autofunc.Result) autofunc.Result {
		return autofunc.Add(inPool, r.Network.Apply(inPool))
	})
}

// ApplyR applies the layer.
func (r *ResidualLayer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return autofunc.PoolR(in, func(inPool autofunc.RResult) autofunc.RResult {
		return autofunc.AddR(inPool, r.Network.ApplyR(rv, inPool))
	})
}

// Batch applies the layer in batch.
func (r *ResidualLayer) Batch(in autofunc.Result, n int) autofunc.Result {
	b := r.Network.BatchLearner()
	return autofunc.Pool(in, func(inPool autofunc.Result) autofunc.Result {
		return autofunc.Add(inPool, b.Batch(inPool, n))
	})
}

// BatchR applies the layer in batch.
func (r *ResidualLayer) BatchR(rv autofunc.RVector, in autofunc.RResult, n int) autofunc.RResult {
	b := r.Network.BatchLearner()
	return autofunc.PoolR(in, func(inPool autofunc.RResult) autofunc.RResult {
		return autofunc.AddR(inPool, b.BatchR(rv, inPool, n))
	})
}

// Parameters returns the parameters of the network.
func (r *ResidualLayer) Parameters() []*autofunc.Variable {
	return r.Network.Parameters()
}

// SerializerType returns the unique ID used to serialize
// a ResidualLayer with the serializer package.
func (r *ResidualLayer) SerializerType() string {
	return serializerTypeResidualLayer
}

// Serialize serializes the layer.
func (r *ResidualLayer) Serialize() ([]byte, error) {
	return serializer.SerializeAny(r.Network)
}
