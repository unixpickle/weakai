package neuralnet

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
)

// A ResidualLayer wraps another layer and adds its input
// the layer's output.
type ResidualLayer struct {
	Layer Layer
}

// DeserializeResidualLayer deserializes a ResidualLayer.
func DeserializeResidualLayer(d []byte) (*ResidualLayer, error) {
	var l Layer
	if err := serializer.DeserializeAny(d, &l); err != nil {
		return nil, err
	}
	return &ResidualLayer{Layer: l}, nil
}

// Apply applies the layer.
func (r *ResidualLayer) Apply(in autofunc.Result) autofunc.Result {
	return autofunc.Pool(in, func(inPool autofunc.Result) autofunc.Result {
		return autofunc.Add(inPool, r.Layer.Apply(inPool))
	})
}

// ApplyR applies the layer.
func (r *ResidualLayer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return autofunc.PoolR(in, func(inPool autofunc.RResult) autofunc.RResult {
		return autofunc.AddR(inPool, r.Layer.ApplyR(rv, inPool))
	})
}

// Batch applies the layer in batch, using the underlying
// Layer as an autofunc.Batcher if possible.
func (r *ResidualLayer) Batch(in autofunc.Result, n int) autofunc.Result {
	b, ok := r.Layer.(autofunc.Batcher)
	if !ok {
		b = &autofunc.FuncBatcher{F: r.Layer}
	}
	return autofunc.Pool(in, func(inPool autofunc.Result) autofunc.Result {
		return autofunc.Add(inPool, b.Batch(inPool, n))
	})
}

// BatchR applies the layer in batch, using the underlying
// Layer as an autofunc.RBatcher if possible.
func (r *ResidualLayer) BatchR(rv autofunc.RVector, in autofunc.RResult, n int) autofunc.RResult {
	b, ok := r.Layer.(autofunc.RBatcher)
	if !ok {
		b = &autofunc.RFuncBatcher{F: r.Layer}
	}
	return autofunc.PoolR(in, func(inPool autofunc.RResult) autofunc.RResult {
		return autofunc.AddR(inPool, b.BatchR(rv, inPool, n))
	})
}

// Parameters returns the parameters of the layer if it is
// an sgd.Learner, or nil otherwise.
func (r *ResidualLayer) Parameters() []*autofunc.Variable {
	if l, ok := r.Layer.(sgd.Learner); ok {
		return l.Parameters()
	}
	return nil
}

// SerializerType returns the unique ID used to serialize
// a ResidualLayer with the serializer package.
func (r *ResidualLayer) SerializerType() string {
	return serializerTypeResidualLayer
}

// Serialize serializes the layer.
func (r *ResidualLayer) Serialize() ([]byte, error) {
	return serializer.SerializeAny(r.Layer)
}
