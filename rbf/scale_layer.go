package rbf

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

func init() {
	var s ScaleLayer
	serializer.RegisterTypedDeserializer(s.SerializerType(), DeserializeScaleLayer)
}

// A ScaleLayer scales its inputs by a learned constant.
// More accurately, the inputs are scaled by the negative
// absolute value of the learned constant.
//
// The scales can be learned separately for each input
// component or can be shared between all components.
type ScaleLayer struct {
	// Has one component for shared scale.
	scale *autofunc.Variable
}

// DeserializeScaleLayer deserializes a ScaleLayer.
func DeserializeScaleLayer(d []byte) (*ScaleLayer, error) {
	if scale, err := autofunc.DeserializeVariable(d); err != nil {
		return nil, err
	} else {
		return &ScaleLayer{scale: scale}, nil
	}
}

// NewScaleLayer creates a ScaleLayer with a different
// learnable scale for each component of the input.
//
// The n argument specifies the input vector size.
//
// The scale argument specifies the initial scale.
func NewScaleLayer(n int, scale float64) *ScaleLayer {
	vec := make(linalg.Vector, n)
	for i := range vec {
		vec[i] = scale
	}
	return &ScaleLayer{
		scale: &autofunc.Variable{Vector: vec},
	}
}

// NewScaleLayerShared creates a ScaleLayer with a sinlge
// scale that is used for every input component.
// The scale argument is used to initialize the scale, but
// learning can change the scale.
func NewScaleLayerShared(scale float64) *ScaleLayer {
	return &ScaleLayer{
		scale: &autofunc.Variable{Vector: []float64{scale}},
	}
}

// Apply applies the layer to an input.
// If the scale is not shared, then the input must have
// the number of components specified when initializing
// the ScaleLayer.
func (s *ScaleLayer) Apply(in autofunc.Result) autofunc.Result {
	if len(s.scale.Vector) == 1 {
		return autofunc.ScaleFirst(in, s.scale)
	} else {
		if len(in.Output()) != len(s.scale.Vector) {
			panic("unexpected component count")
		}
		return autofunc.Mul(in, s.scale)
	}
}

// ApplyR is like Apply but for RResults.
func (s *ScaleLayer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	sc := autofunc.NewRVariable(s.scale, rv)
	if len(s.scale.Vector) == 1 {
		return autofunc.ScaleFirstR(in, sc)
	} else {
		if len(in.Output()) != len(s.scale.Vector) {
			panic("unexpected component count")
		}
		return autofunc.MulR(in, sc)
	}
}

// Parameters returns a slice containing one variable for
// the component scale.
// If the scale is shared, then the variable will have one
// component.
func (s *ScaleLayer) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{s.scale}
}

// SerializerType returns the unique ID used to serialize
// a ScaleLayer.
func (s *ScaleLayer) SerializerType() string {
	return "github.com/unixpickle/weakai/rbf.ScaleLayer"
}

// Serialize serializes the layer.
func (s *ScaleLayer) Serialize() ([]byte, error) {
	return s.scale.Serialize()
}
