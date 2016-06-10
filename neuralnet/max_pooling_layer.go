package neuralnet

import (
	"encoding/json"
	"math"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A MaxPoolingLayer reduces the width and height
// of an input tensor by returning the maximum value
// from each of many small two-dimensional regions
// in each depth layer of the input tensor.
type MaxPoolingLayer struct {
	// XSpan indicates how many consecutive
	// horizontal inputs correspond to a pool.
	XSpan int

	// YSpan indicates how many consecutive
	// vertical inputs correspond to a pool.
	YSpan int

	// InputWidth indicates the width of the
	// layer's input tensor.
	InputWidth int

	// InputHeight indicates the height of the
	// layer's input tensor.
	InputHeight int

	// InputDepth indicates the depth of the
	// layer's input tensor.
	InputDepth int

	Cache *autofunc.VectorCache
}

func DeserializeMaxPoolingLayer(d []byte) (*MaxPoolingLayer, error) {
	var res MaxPoolingLayer
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

func (m *MaxPoolingLayer) OutputWidth() int {
	w := m.InputWidth / m.XSpan
	if (m.InputWidth % m.XSpan) != 0 {
		w++
	}
	return w
}

func (m *MaxPoolingLayer) OutputHeight() int {
	h := m.InputHeight / m.YSpan
	if (m.InputHeight % m.YSpan) != 0 {
		h++
	}
	return h
}

func (m *MaxPoolingLayer) Apply(in autofunc.Result) autofunc.Result {
	inTensor := m.inputTensor(in.Output())
	out, choices := m.evaluate(inTensor)
	return &maxPoolingResult{
		OutputTensor: out,
		Choices:      choices,
		Input:        in,
		Layer:        m,
	}
}

func (m *MaxPoolingLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	inTensor := m.inputTensor(in.Output())
	inTensorR := m.inputTensor(in.ROutput())
	out, choices := m.evaluate(inTensor)
	outR := choices.ForwardPropagate(m.Cache, inTensorR)
	return &maxPoolingRResult{
		OutputTensor:  out,
		ROutputTensor: outR,
		Choices:       choices,
		Input:         in,
		Layer:         m,
	}
}

func (m *MaxPoolingLayer) SetCache(c *autofunc.VectorCache) {
	m.Cache = c
}

func (m *MaxPoolingLayer) Serialize() ([]byte, error) {
	return json.Marshal(m)
}

func (m *MaxPoolingLayer) SerializerType() string {
	return serializerTypeMaxPoolingLayer
}

func (m *MaxPoolingLayer) evaluate(inTensor *Tensor3) (*Tensor3, poolChoiceMap) {
	outTensor := NewTensor3Cache(m.Cache, m.OutputWidth(), m.OutputHeight(), m.InputDepth)
	choices := newPoolChoiceMap(m.OutputWidth(), m.OutputHeight(), m.InputDepth)
	for y := 0; y < outTensor.Height; y++ {
		poolY := y * m.YSpan
		maxY := poolY + m.YSpan - 1
		if maxY >= inTensor.Height {
			maxY = inTensor.Height - 1
		}
		for x := 0; x < outTensor.Width; x++ {
			poolX := x * m.XSpan
			maxX := poolX + m.XSpan - 1
			if maxX >= inTensor.Width {
				maxX = inTensor.Width - 1
			}
			for z := 0; z < outTensor.Depth; z++ {
				output, bestX, bestY := maxInput(inTensor, poolX, maxX, poolY, maxY, z)
				outTensor.Set(x, y, z, output)
				choices[y][x][z] = [2]int{bestX, bestY}
			}
		}
	}
	return outTensor, choices
}

func (m *MaxPoolingLayer) inputTensor(inVec linalg.Vector) *Tensor3 {
	return &Tensor3{
		Width:  m.InputWidth,
		Height: m.InputHeight,
		Depth:  m.InputDepth,
		Data:   inVec,
	}
}

func (m *MaxPoolingLayer) outputTensor(outVec linalg.Vector) *Tensor3 {
	return &Tensor3{
		Width:  m.OutputWidth(),
		Height: m.OutputHeight(),
		Depth:  m.InputDepth,
		Data:   outVec,
	}
}

type maxPoolingResult struct {
	OutputTensor *Tensor3
	Choices      poolChoiceMap
	Input        autofunc.Result
	Layer        *MaxPoolingLayer
}

func (m *maxPoolingResult) Output() linalg.Vector {
	return m.OutputTensor.Data
}

func (m *maxPoolingResult) Constant(g autofunc.Gradient) bool {
	return m.Input.Constant(g)
}

func (m *maxPoolingResult) PropagateGradient(upstream linalg.Vector, grad autofunc.Gradient) {
	if m.Input.Constant(grad) {
		return
	}
	ut := m.Layer.outputTensor(upstream)
	downstream := m.Choices.BackPropagate(m.Layer.Cache, ut, m.Layer.InputWidth,
		m.Layer.InputHeight)
	m.Input.PropagateGradient(downstream.Data, grad)
	m.Layer.Cache.Free(downstream.Data)
}

func (m *maxPoolingResult) Release() {
	m.Layer.Cache.Free(m.OutputTensor.Data)
	m.OutputTensor.Data = nil
	m.Input.Release()
}

type maxPoolingRResult struct {
	OutputTensor  *Tensor3
	ROutputTensor *Tensor3
	Choices       poolChoiceMap
	Input         autofunc.RResult
	Layer         *MaxPoolingLayer
}

func (m *maxPoolingRResult) Output() linalg.Vector {
	return m.OutputTensor.Data
}

func (m *maxPoolingRResult) ROutput() linalg.Vector {
	return m.ROutputTensor.Data
}

func (m *maxPoolingRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return m.Input.Constant(rg, g)
}

func (m *maxPoolingRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad autofunc.RGradient, grad autofunc.Gradient) {
	if m.Input.Constant(rgrad, grad) {
		return
	}
	ut := m.Layer.outputTensor(upstream)
	utR := m.Layer.outputTensor(upstreamR)
	downstream := m.Choices.BackPropagate(m.Layer.Cache, ut, m.Layer.InputWidth,
		m.Layer.InputHeight)
	downstreamR := m.Choices.BackPropagate(m.Layer.Cache, utR, m.Layer.InputWidth,
		m.Layer.InputHeight)
	m.Input.PropagateRGradient(downstream.Data, downstreamR.Data, rgrad, grad)
	m.Layer.Cache.Free(downstream.Data)
	m.Layer.Cache.Free(downstreamR.Data)
}

func (m *maxPoolingRResult) Release() {
	m.Layer.Cache.Free(m.OutputTensor.Data)
	m.Layer.Cache.Free(m.ROutputTensor.Data)
	m.OutputTensor.Data = nil
	m.ROutputTensor.Data = nil
	m.Input.Release()
}

func maxInput(t *Tensor3, x1, x2, y1, y2, z int) (value float64, bestX, bestY int) {
	value = math.Inf(-1)
	for x := x1; x <= x2; x++ {
		for y := y1; y <= y2; y++ {
			input := t.Get(x, y, z)
			if input > value {
				value = input
				bestX = x
				bestY = y
			}
		}
	}
	return
}

// poolChoiceMap maps each point in an output tensor
// to a point in an input tensor.
type poolChoiceMap [][][][2]int

func newPoolChoiceMap(width, height, depth int) poolChoiceMap {
	res := make(poolChoiceMap, height)
	for i := range res {
		res[i] = make([][][2]int, width)
		for j := range res[i] {
			res[i][j] = make([][2]int, depth)
		}
	}
	return res
}

func (p poolChoiceMap) ForwardPropagate(c *autofunc.VectorCache, in *Tensor3) *Tensor3 {
	output := NewTensor3Cache(c, p.Width(), p.Height(), in.Depth)
	for y, list := range p {
		for x, list1 := range list {
			for z, point := range list1 {
				val := in.Get(point[0], point[1], z)
				output.Set(x, y, z, val)
			}
		}
	}
	return output
}

func (p poolChoiceMap) BackPropagate(c *autofunc.VectorCache, in *Tensor3,
	outWidth, outHeight int) *Tensor3 {
	output := NewTensor3Cache(c, outWidth, outHeight, in.Depth)
	for y, list := range p {
		for x, list1 := range list {
			for z, point := range list1 {
				val := in.Get(x, y, z)
				output.Set(point[0], point[1], z, val)
			}
		}
	}
	return output
}

func (p poolChoiceMap) Width() int {
	if len(p) == 0 {
		return 0
	}
	return len(p[0])
}

func (p poolChoiceMap) Height() int {
	return len(p)
}
