package neuralnet

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// RescaleLayer is a Layer which adds a bias to its
// input and scales the translated input.
// It is useful for ensuring that input samples have
// a mean of 0 and a standard deviation of 1.
type RescaleLayer struct {
	Bias  float64
	Scale float64
}

func DeserializeRescaleLayer(d []byte) (*RescaleLayer, error) {
	var res RescaleLayer
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

func (r *RescaleLayer) Apply(in autofunc.Result) autofunc.Result {
	return autofunc.Scale(autofunc.AddScaler(in, r.Bias), r.Scale)
}

func (r *RescaleLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return autofunc.ScaleR(autofunc.AddScalerR(in, r.Bias), r.Scale)
}

func (r *RescaleLayer) Serialize() ([]byte, error) {
	return json.Marshal(r)
}

func (r *RescaleLayer) SerializerType() string {
	return serializerTypeRescaleLayer
}

// VecRescaleLayer is similar to a RescaleLayer, but
// it applies a different bias and scale to each entry
// of its input vectors.
type VecRescaleLayer struct {
	Biases linalg.Vector
	Scales linalg.Vector
}

func DeserializeVecRescaleLayer(d []byte) (*VecRescaleLayer, error) {
	var res VecRescaleLayer
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

func (v *VecRescaleLayer) Apply(in autofunc.Result) autofunc.Result {
	return autofunc.Mul(autofunc.Add(in, &autofunc.Variable{Vector: v.Biases}),
		&autofunc.Variable{Vector: v.Scales})
}

func (v *VecRescaleLayer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	zeroVec := make(linalg.Vector, len(in.Output()))
	biases := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: v.Biases},
		ROutputVec: zeroVec,
	}
	scales := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: v.Scales},
		ROutputVec: zeroVec,
	}
	return autofunc.MulR(autofunc.AddR(in, biases), scales)
}

func (v *VecRescaleLayer) Serialize() ([]byte, error) {
	return json.Marshal(v)
}

func (v *VecRescaleLayer) SerializerType() string {
	return serializerTypeVecRescaleLayer
}
