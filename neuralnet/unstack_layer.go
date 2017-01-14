package neuralnet

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/tensor"
)

// UnstackLayer unstacks input tensors.
// It is useful for enlarging initially small images.
//
// As an analogy, imagine taking a stack of 16 sheets
// of paper and rearranging it into a 2x2 grid of
// stacks of four sheets of paper each.
// In this analogy, 2 is the InverseStride, 4 is the
// new output depth, and 16 is the input depth.
type UnstackLayer struct {
	InputWidth  int
	InputHeight int
	InputDepth  int

	// InverseStride is the side length of the new
	// rectangular regions which will be formed by
	// unstacking input tensors.
	// The square of this value must divide InputDepth.
	InverseStride int
}

func DeserializeUnstackLayer(d []byte) (*UnstackLayer, error) {
	var res UnstackLayer
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

func (u *UnstackLayer) Apply(in autofunc.Result) autofunc.Result {
	return &unstackLayerResult{
		OutputVector: u.unstack(in.Output()),
		Input:        in,
		Layer:        u,
	}
}

func (u *UnstackLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return &unstackLayerRResult{
		OutputVector:  u.unstack(in.Output()),
		ROutputVector: u.unstack(in.ROutput()),
		Input:         in,
		Layer:         u,
	}
}

func (u *UnstackLayer) Serialize() ([]byte, error) {
	return json.Marshal(u)
}

func (u *UnstackLayer) SerializerType() string {
	return serializerTypeUnstackLayer
}

func (u *UnstackLayer) unstack(inVec linalg.Vector) linalg.Vector {
	if u.InputDepth%(u.InverseStride*u.InverseStride) != 0 {
		panic("InverseStride^2 must divide InputDepth")
	}

	input := &tensor.Float64{
		Width:  u.InputWidth,
		Height: u.InputHeight,
		Depth:  u.InputDepth,
		Data:   inVec,
	}
	output := tensor.NewFloat64(u.InputWidth*u.InverseStride,
		u.InputHeight*u.InverseStride,
		u.InputDepth/(u.InverseStride*u.InverseStride))

	for y := 0; y < output.Height; y++ {
		internalY := y / u.InverseStride
		for x := 0; x < output.Width; x++ {
			internalX := x / u.InverseStride
			internalOffset := ((x % u.InverseStride) + u.InverseStride*(y%u.InverseStride)) *
				output.Depth
			for z := 0; z < output.Depth; z++ {
				val := input.Get(internalX, internalY, internalOffset+z)
				output.Set(x, y, z, val)
			}
		}
	}

	return output.Data
}

func (u *UnstackLayer) stack(inVec linalg.Vector) linalg.Vector {
	unstacked := &tensor.Float64{
		Width:  u.InputWidth * u.InverseStride,
		Height: u.InputHeight * u.InverseStride,
		Depth:  u.InputDepth / (u.InverseStride * u.InverseStride),
		Data:   inVec,
	}

	stacked := tensor.NewFloat64(u.InputWidth, u.InputHeight, u.InputDepth)

	for y := 0; y < stacked.Height; y++ {
		unstackedY := y * u.InverseStride
		for x := 0; x < stacked.Width; x++ {
			unstackedX := x * u.InverseStride
			for z := 0; z < stacked.Depth; z++ {
				internalZ := z % unstacked.Depth
				dived := z / unstacked.Depth
				internalX := dived % u.InverseStride
				internalY := dived / u.InverseStride

				value := unstacked.Get(unstackedX+internalX, unstackedY+internalY, internalZ)
				stacked.Set(x, y, z, value)
			}
		}
	}

	return stacked.Data
}

type unstackLayerResult struct {
	OutputVector linalg.Vector
	Input        autofunc.Result
	Layer        *UnstackLayer
}

func (u *unstackLayerResult) Output() linalg.Vector {
	return u.OutputVector
}

func (u *unstackLayerResult) Constant(g autofunc.Gradient) bool {
	return u.Input.Constant(g)
}

func (u *unstackLayerResult) PropagateGradient(upstream linalg.Vector, grad autofunc.Gradient) {
	if !u.Input.Constant(grad) {
		downstream := u.Layer.stack(upstream)
		u.Input.PropagateGradient(downstream, grad)
	}
}

type unstackLayerRResult struct {
	OutputVector  linalg.Vector
	ROutputVector linalg.Vector
	Input         autofunc.RResult
	Layer         *UnstackLayer
}

func (u *unstackLayerRResult) Output() linalg.Vector {
	return u.OutputVector
}

func (u *unstackLayerRResult) ROutput() linalg.Vector {
	return u.ROutputVector
}

func (u *unstackLayerRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return u.Input.Constant(rg, g)
}

func (u *unstackLayerRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad autofunc.RGradient, grad autofunc.Gradient) {
	if !u.Input.Constant(rgrad, grad) {
		downstream := u.Layer.stack(upstream)
		downstreamR := u.Layer.stack(upstreamR)
		u.Input.PropagateRGradient(downstream, downstreamR, rgrad, grad)
	}
}
