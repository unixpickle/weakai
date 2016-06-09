package neuralnet

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// BorderLayer stores parameters used
// to configure a BorderLayer.
type BorderLayer struct {
	InputWidth  int
	InputHeight int
	InputDepth  int

	LeftBorder   int
	RightBorder  int
	TopBorder    int
	BottomBorder int

	Cache *autofunc.VectorCache
}

func DeserializeBorderLayer(data []byte) (*BorderLayer, error) {
	var s BorderLayer
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

func (b *BorderLayer) Apply(in autofunc.Result) autofunc.Result {
	return &borderResult{
		OutputVec: b.addBorder(in.Output()),
		Input:     in,
		Info:      b,
	}
}

func (b *BorderLayer) ApplyR(in autofunc.RResult) autofunc.RResult {
	return &borderRResult{
		OutputVec:  b.addBorder(in.Output()),
		ROutputVec: b.addBorder(in.ROutput()),
		Input:      in,
		Info:       b,
	}
}

func (b *BorderLayer) SetCache(c *autofunc.VectorCache) {
	b.Cache = c
}

func (b *BorderLayer) Serialize() ([]byte, error) {
	return json.Marshal(b)
}

func (b *BorderLayer) SerializerType() string {
	return serializerTypeBorderLayer
}

func (b *BorderLayer) addBorder(tensorVec linalg.Vector) linalg.Vector {
	inTensor := &Tensor3{
		Width:  b.InputWidth,
		Height: b.InputHeight,
		Depth:  b.InputDepth,
		Data:   tensorVec,
	}
	outTensor := NewTensor3Cache(b.Cache, b.InputWidth+b.LeftBorder+b.RightBorder,
		b.InputHeight+b.TopBorder+b.BottomBorder, b.InputDepth)
	for y := 0; y < inTensor.Height; y++ {
		insetY := y + b.TopBorder
		for x := 0; x < inTensor.Width; x++ {
			insetX := x + b.LeftBorder
			for z := 0; z < inTensor.Depth; z++ {
				inVal := inTensor.Get(x, y, z)
				outTensor.Set(insetX, insetY, z, inVal)
			}
		}
	}
	return outTensor.Data
}

func (b *BorderLayer) removeBorder(tensorVec linalg.Vector) linalg.Vector {
	outTensor := NewTensor3Cache(b.Cache, b.InputWidth, b.InputHeight, b.InputDepth)
	inTensor := &Tensor3{
		Width:  b.InputWidth + b.LeftBorder + b.RightBorder,
		Height: b.InputHeight + b.TopBorder + b.BottomBorder,
		Depth:  b.InputDepth,
		Data:   tensorVec,
	}
	for y := 0; y < outTensor.Height; y++ {
		insetY := y + b.TopBorder
		for x := 0; x < outTensor.Width; x++ {
			insetX := x + b.LeftBorder
			for z := 0; z < outTensor.Depth; z++ {
				inVal := inTensor.Get(insetX, insetY, z)
				outTensor.Set(x, y, z, inVal)
			}
		}
	}
	return outTensor.Data
}

type borderResult struct {
	OutputVec linalg.Vector
	Input     autofunc.Result
	Info      *BorderLayer
}

func (b *borderResult) Output() linalg.Vector {
	return b.OutputVec
}

func (b *borderResult) Constant(g autofunc.Gradient) bool {
	return b.Input.Constant(g)
}

func (b *borderResult) PropagateGradient(upstream linalg.Vector, grad autofunc.Gradient) {
	if !b.Input.Constant(grad) {
		downstream := b.Info.removeBorder(upstream)
		b.Input.PropagateGradient(downstream, grad)
	}
}

func (b *borderResult) Release() {
	b.Info.Cache.Free(b.OutputVec)
	b.OutputVec = nil
	b.Input.Release()
}

type borderRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      autofunc.RResult
	Info       *BorderLayer
}

func (b *borderRResult) Output() linalg.Vector {
	return b.OutputVec
}

func (b *borderRResult) ROutput() linalg.Vector {
	return b.ROutputVec
}

func (b *borderRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return b.Input.Constant(rg, g)
}

func (b *borderRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad autofunc.RGradient, grad autofunc.Gradient) {
	if !b.Input.Constant(rgrad, grad) {
		downstream := b.Info.removeBorder(upstream)
		downstreamR := b.Info.removeBorder(upstreamR)
		b.Input.PropagateRGradient(downstream, downstreamR, rgrad, grad)
	}
}

func (b *borderRResult) Release() {
	b.Info.Cache.Free(b.OutputVec)
	b.Info.Cache.Free(b.ROutputVec)
	b.OutputVec = nil
	b.ROutputVec = nil
	b.Input.Release()
}
