package neuralnet

import (
	"encoding/json"
	"math"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/tensor"
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
}

// DeserializeMaxPoolingLayer deserializes a MaxPoolingLayer.
func DeserializeMaxPoolingLayer(d []byte) (*MaxPoolingLayer, error) {
	var res MaxPoolingLayer
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// OutputWidth returns the output tensor width.
func (m *MaxPoolingLayer) OutputWidth() int {
	w := m.InputWidth / m.XSpan
	if (m.InputWidth % m.XSpan) != 0 {
		w++
	}
	return w
}

// OutputHeight returns the output tensor height.
func (m *MaxPoolingLayer) OutputHeight() int {
	h := m.InputHeight / m.YSpan
	if (m.InputHeight % m.YSpan) != 0 {
		h++
	}
	return h
}

// Apply applies the layer to an input, which is treated
// as a tensor.
func (m *MaxPoolingLayer) Apply(in autofunc.Result) autofunc.Result {
	return m.Batch(in, 1)
}

// ApplyR is like Apply, but for RResults.
func (m *MaxPoolingLayer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return m.BatchR(rv, in, 1)
}

// Batch applies the layer to inputs in batch.
func (m *MaxPoolingLayer) Batch(in autofunc.Result, n int) autofunc.Result {
	outSize := m.OutputWidth() * m.OutputHeight() * m.InputDepth
	inSize := m.InputWidth * m.InputHeight * m.InputDepth
	if len(in.Output()) != n*inSize {
		panic("invalid input size")
	}
	res := &maxPoolingResult{
		OutputVec: make(linalg.Vector, outSize*n),
		Input:     in,
		Layer:     m,
	}
	for i := 0; i < n; i++ {
		outTensor := m.outputTensor(res.OutputVec[i*outSize : (i+1)*outSize])
		inTensor := m.inputTensor(in.Output()[i*inSize : (i+1)*inSize])
		choices := m.evaluate(inTensor, outTensor)
		res.Choices = append(res.Choices, choices)
	}
	return res
}

// BatchR is like Batch, but for RResults.
func (m *MaxPoolingLayer) BatchR(rv autofunc.RVector, in autofunc.RResult,
	n int) autofunc.RResult {
	outSize := m.OutputWidth() * m.OutputHeight() * m.InputDepth
	inSize := m.InputWidth * m.InputHeight * m.InputDepth
	if len(in.Output()) != n*inSize {
		panic("invalid input size")
	}
	res := &maxPoolingRResult{
		OutputVec:  make(linalg.Vector, outSize*n),
		ROutputVec: make(linalg.Vector, outSize*n),
		Input:      in,
		Layer:      m,
	}
	for i := 0; i < n; i++ {
		outTensor := m.outputTensor(res.OutputVec[i*outSize : (i+1)*outSize])
		inTensor := m.inputTensor(in.Output()[i*inSize : (i+1)*inSize])
		choices := m.evaluate(inTensor, outTensor)
		res.Choices = append(res.Choices, choices)

		outTensorR := m.outputTensor(res.ROutputVec[i*outSize : (i+1)*outSize])
		inTensorR := m.inputTensor(in.ROutput()[i*inSize : (i+1)*inSize])
		choices.ForwardPropagate(inTensorR, outTensorR)
	}
	return res
}

// Serialize serializes the layer.
func (m *MaxPoolingLayer) Serialize() ([]byte, error) {
	return json.Marshal(m)
}

// SerializerType returns the unique ID used to serialize
// this layer with the serializer package.
func (m *MaxPoolingLayer) SerializerType() string {
	return serializerTypeMaxPoolingLayer
}

func (m *MaxPoolingLayer) evaluate(in *tensor.Float64, out *tensor.Float64) poolChoiceMap {
	choices := newPoolChoiceMap(m.OutputWidth(), m.OutputHeight(), m.InputDepth)
	for y := 0; y < out.Height; y++ {
		poolY := y * m.YSpan
		maxY := poolY + m.YSpan - 1
		if maxY >= in.Height {
			maxY = in.Height - 1
		}
		for x := 0; x < out.Width; x++ {
			poolX := x * m.XSpan
			maxX := poolX + m.XSpan - 1
			if maxX >= in.Width {
				maxX = in.Width - 1
			}
			for z := 0; z < out.Depth; z++ {
				output, bestX, bestY := maxInput(in, poolX, maxX, poolY, maxY, z)
				out.Set(x, y, z, output)
				choices[y][x][z] = [2]int{bestX, bestY}
			}
		}
	}
	return choices
}

func (m *MaxPoolingLayer) inputTensor(inVec linalg.Vector) *tensor.Float64 {
	return &tensor.Float64{
		Width:  m.InputWidth,
		Height: m.InputHeight,
		Depth:  m.InputDepth,
		Data:   inVec,
	}
}

func (m *MaxPoolingLayer) outputTensor(outVec linalg.Vector) *tensor.Float64 {
	return &tensor.Float64{
		Width:  m.OutputWidth(),
		Height: m.OutputHeight(),
		Depth:  m.InputDepth,
		Data:   outVec,
	}
}

type maxPoolingResult struct {
	OutputVec linalg.Vector
	Choices   []poolChoiceMap
	Input     autofunc.Result
	Layer     *MaxPoolingLayer
}

func (m *maxPoolingResult) Output() linalg.Vector {
	return m.OutputVec
}

func (m *maxPoolingResult) Constant(g autofunc.Gradient) bool {
	return m.Input.Constant(g)
}

func (m *maxPoolingResult) PropagateGradient(upstream linalg.Vector, grad autofunc.Gradient) {
	if m.Input.Constant(grad) {
		return
	}
	downstream := make(linalg.Vector, len(m.Input.Output()))
	subUpstreamSize := len(m.OutputVec) / len(m.Choices)
	subDownstreamSize := len(downstream) / len(m.Choices)
	for i, choices := range m.Choices {
		subUp := upstream[i*subUpstreamSize : (i+1)*subUpstreamSize]
		subDown := downstream[i*subDownstreamSize : (i+1)*subDownstreamSize]
		choices.BackPropagate(m.Layer.outputTensor(subUp),
			m.Layer.inputTensor(subDown))
	}
	m.Input.PropagateGradient(downstream, grad)
}

type maxPoolingRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Choices    []poolChoiceMap
	Input      autofunc.RResult
	Layer      *MaxPoolingLayer
}

func (m *maxPoolingRResult) Output() linalg.Vector {
	return m.OutputVec
}

func (m *maxPoolingRResult) ROutput() linalg.Vector {
	return m.ROutputVec
}

func (m *maxPoolingRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return m.Input.Constant(rg, g)
}

func (m *maxPoolingRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad autofunc.RGradient, grad autofunc.Gradient) {
	if m.Input.Constant(rgrad, grad) {
		return
	}
	downstream := make(linalg.Vector, len(m.Input.Output()))
	downstreamR := make(linalg.Vector, len(m.Input.Output()))
	subUpstreamSize := len(m.OutputVec) / len(m.Choices)
	subDownstreamSize := len(downstream) / len(m.Choices)
	for i, choices := range m.Choices {
		subUp := upstream[i*subUpstreamSize : (i+1)*subUpstreamSize]
		subDown := downstream[i*subDownstreamSize : (i+1)*subDownstreamSize]
		choices.BackPropagate(m.Layer.outputTensor(subUp),
			m.Layer.inputTensor(subDown))

		subUpR := upstreamR[i*subUpstreamSize : (i+1)*subUpstreamSize]
		subDownR := downstreamR[i*subDownstreamSize : (i+1)*subDownstreamSize]
		choices.BackPropagate(m.Layer.outputTensor(subUpR),
			m.Layer.inputTensor(subDownR))
	}
	m.Input.PropagateRGradient(downstream, downstreamR, rgrad, grad)
}

func maxInput(t *tensor.Float64, x1, x2, y1, y2, z int) (value float64, bestX, bestY int) {
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

func (p poolChoiceMap) ForwardPropagate(in *tensor.Float64, out *tensor.Float64) {
	for y, list := range p {
		for x, list1 := range list {
			for z, point := range list1 {
				val := in.Get(point[0], point[1], z)
				out.Set(x, y, z, val)
			}
		}
	}
}

func (p poolChoiceMap) BackPropagate(downstream *tensor.Float64, upstream *tensor.Float64) {
	for y, list := range p {
		for x, list1 := range list {
			for z, point := range list1 {
				val := downstream.Get(x, y, z)
				upstream.Set(point[0], point[1], z, val)
			}
		}
	}
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
