package rbf

import (
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

func init() {
	var d DistLayer
	serializer.RegisterTypedDeserializer(d.SerializerType(), DeserializeDistLayer)
}

// DistLayer computes the squared distances of the input
// vector from a list of "centers".
type DistLayer struct {
	outputCount int
	centers     *autofunc.Variable
}

// DeserializeDistLayer deserializes a DistLayer.
func DeserializeDistLayer(d []byte) (*DistLayer, error) {
	var num serializer.Int
	var res DistLayer
	if err := serializer.DeserializeAny(d, &num, &res.centers); err != nil {
		return nil, err
	}
	res.outputCount = int(num)
	return &res, nil
}

// NewDistLayer creates a layer with centers whose
// components are sampled from a normal random variable,
// scaled by scale.
func NewDistLayer(inCount, outCount int, scale float64) *DistLayer {
	res := &DistLayer{
		outputCount: outCount,
		centers: &autofunc.Variable{
			Vector: make(linalg.Vector, inCount*outCount),
		},
	}
	for i := range res.centers.Vector {
		res.centers.Vector[i] = rand.NormFloat64() * scale
	}
	return res
}

// Apply applies the layer to an input vector.
func (d *DistLayer) Apply(in autofunc.Result) autofunc.Result {
	return d.Batch(in, 1)
}

// ApplyR applies the layer to an input vector.
func (d *DistLayer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return d.BatchR(rv, in, 1)
}

// Batch applies the layer in batch.
func (d *DistLayer) Batch(in autofunc.Result, n int) autofunc.Result {
	return autofunc.Pool(in, func(in autofunc.Result) autofunc.Result {
		ins := autofunc.Split(n, in)
		centers := autofunc.Split(d.NumCenters(), d.centers)
		comps := make([][]autofunc.Result, n)
		for _, x := range centers {
			neg := autofunc.Scale(x, -1)
			for j, in := range ins {
				dist := autofunc.SquaredNorm{}.Apply(autofunc.Add(in, neg))
				comps[j] = append(comps[j], dist)
			}
		}
		var outComps []autofunc.Result
		for _, x := range comps {
			for _, y := range x {
				outComps = append(outComps, y)
			}
		}
		return autofunc.Concat(outComps...)
	})
}

// BatchR applies the layer in batch.
func (d *DistLayer) BatchR(rv autofunc.RVector, in autofunc.RResult, n int) autofunc.RResult {
	return autofunc.PoolR(in, func(in autofunc.RResult) autofunc.RResult {
		ins := autofunc.SplitR(n, in)
		centers := autofunc.SplitR(d.NumCenters(), autofunc.NewRVariable(d.centers, rv))
		comps := make([][]autofunc.RResult, n)
		for _, x := range centers {
			neg := autofunc.ScaleR(x, -1)
			for j, in := range ins {
				dist := autofunc.SquaredNorm{}.ApplyR(rv, autofunc.AddR(in, neg))
				comps[j] = append(comps[j], dist)
			}
		}
		var outComps []autofunc.RResult
		for _, x := range comps {
			for _, y := range x {
				outComps = append(outComps, y)
			}
		}
		return autofunc.ConcatR(outComps...)
	})
}

// NumCenters returns the number of centers (and thus
// the number of outputs).
func (d *DistLayer) NumCenters() int {
	return d.outputCount
}

// SetCenters updates the centers of the network.
func (d *DistLayer) SetCenters(c []linalg.Vector) {
	if len(c) != d.NumCenters() {
		panic("bad number of centers")
	}
	inSize := len(d.centers.Vector) / d.outputCount
	for i, x := range c {
		if len(x) != inSize {
			panic("invalid center size")
		}
		copy(d.centers.Vector[inSize*i:inSize*(i+1)], x)
	}
}

// Parameters returns a slice containing one variable: the
// matrix of centers (each row is a center).
func (d *DistLayer) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{d.centers}
}

// SerializerType returns the unique ID used to serialize
// a DistLayer with the serializer package.
func (d *DistLayer) SerializerType() string {
	return "github.com/unixpickle/weakai/rbf.DistLayer"
}

// Serialize serializes the layer.
func (d *DistLayer) Serialize() ([]byte, error) {
	return serializer.SerializeAny(serializer.Int(d.outputCount), d.centers)
}
