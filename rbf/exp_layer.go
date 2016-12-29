package rbf

import (
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
)

func init() {
	var e ExpLayer
	serializer.RegisterTypedDeserializer(e.SerializerType(), DeserializeExpLayer)
}

// ExpLayer exponentiates its inputs, with the option to
// normalize the results by dividing by the sum.
type ExpLayer struct {
	Normalize bool
}

// DeserializeExpLayer deserializes an ExpLayer.
func DeserializeExpLayer(d []byte) (*ExpLayer, error) {
	if len(d) != 1 {
		return nil, errors.New("expected 1 byte for ExpLayer")
	}
	return &ExpLayer{Normalize: d[0] != 0}, nil
}

// Apply applies the layer to an input.
func (e *ExpLayer) Apply(in autofunc.Result) autofunc.Result {
	if !e.Normalize {
		return autofunc.Exp{}.Apply(in)
	}
	return autofunc.Pool(in, func(in autofunc.Result) autofunc.Result {
		scale := autofunc.Scale(autofunc.SumAllLogDomain(in), -1)
		return autofunc.Exp{}.Apply(autofunc.AddFirst(in, scale))
	})
}

// ApplyR applies the layer to an input.
func (e *ExpLayer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	if !e.Normalize {
		return autofunc.Exp{}.ApplyR(rv, in)
	}
	return autofunc.PoolR(in, func(in autofunc.RResult) autofunc.RResult {
		scale := autofunc.ScaleR(autofunc.SumAllLogDomainR(in), -1)
		return autofunc.Exp{}.ApplyR(rv, autofunc.AddFirstR(in, scale))
	})
}

// SerializerType returns the unique ID used to serialize
// an ExpLayer with the serializer package.
func (e *ExpLayer) SerializerType() string {
	return "github.com/unixpickle/weakai/rbf.ExpLayer"
}

// Serialize serializes the layer.
func (e *ExpLayer) Serialize() ([]byte, error) {
	if e.Normalize {
		return []byte{1}, nil
	}
	return []byte{0}, nil
}
