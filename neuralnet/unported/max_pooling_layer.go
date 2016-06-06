package neuralnet

import (
	"encoding/json"
	"math"
)

// MaxPoolingParams configures a max-pooling
// layer for a neural network.
type MaxPoolingParams struct {
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

// Make creates a new *MaxPoolingLayer using
// the values in p.
// This is equivalent to NewMaxPoolingLayer(p).
func (p *MaxPoolingParams) Make() Layer {
	return NewMaxPoolingLayer(p)
}

type MaxPoolingLayer struct {
	tensorLayer

	xSpan int
	ySpan int

	// outputChoices[x][y][z] is the pair {x1,y1}
	// indicating which value in the pool associated
	// with output (x,y,z) was chosen.
	// The pair {x1,y1,z} is the coordinate in
	// the input tensor which was output.
	outputChoices [][][][2]int
}

func NewMaxPoolingLayer(params *MaxPoolingParams) *MaxPoolingLayer {
	w := params.InputWidth / params.XSpan
	if (params.InputWidth % params.XSpan) != 0 {
		w++
	}
	h := params.InputHeight / params.YSpan
	if (params.InputHeight % params.YSpan) != 0 {
		h++
	}
	res := &MaxPoolingLayer{
		tensorLayer: tensorLayer{
			output:           NewTensor3(w, h, params.InputDepth),
			upstreamGradient: NewTensor3(params.InputWidth, params.InputHeight, params.InputDepth),
		},
		xSpan:         params.XSpan,
		ySpan:         params.YSpan,
		outputChoices: make([][][][2]int, w),
	}
	for i := range res.outputChoices {
		list := make([][][2]int, h)
		for j := range list {
			list[j] = make([][2]int, params.InputDepth)
		}
		res.outputChoices[i] = list
	}
	return res
}

func DeserializeMaxPoolingLayer(data []byte) (*MaxPoolingLayer, error) {
	var s serializedMaxPoolingLayer
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, err
	}

	res := &MaxPoolingLayer{
		tensorLayer: tensorLayer{
			output:           NewTensor3(s.OutputWidth, s.OutputHeight, s.OutputDepth),
			upstreamGradient: NewTensor3(s.InputWidth, s.InputHeight, s.InputDepth),
		},
		xSpan:         s.XSpan,
		ySpan:         s.YSpan,
		outputChoices: make([][][][2]int, s.OutputWidth),
	}

	for i := range res.outputChoices {
		list := make([][][2]int, s.OutputHeight)
		for j := range list {
			list[j] = make([][2]int, s.OutputDepth)
		}
		res.outputChoices[i] = list
	}

	return res, nil
}

// Randomize does nothing, since this type of
// layer has no learnable values.
func (r *MaxPoolingLayer) Randomize() {
}

func (r *MaxPoolingLayer) PropagateForward() {
	for y := 0; y < r.output.Height; y++ {
		poolY := y * r.ySpan
		maxY := poolY + r.ySpan - 1
		if maxY >= r.input.Height {
			maxY = r.input.Height - 1
		}
		for x := 0; x < r.output.Width; x++ {
			poolX := x * r.xSpan
			maxX := poolX + r.xSpan - 1
			if maxX >= r.input.Width {
				maxX = r.input.Width - 1
			}
			for z := 0; z < r.output.Depth; z++ {
				output, bestX, bestY := r.maxInput(poolX, maxX, poolY, maxY, z)
				r.output.Set(x, y, z, output)
				r.outputChoices[x][y][z] = [2]int{bestX, bestY}
			}
		}
	}
}

func (r *MaxPoolingLayer) PropagateBackward(upstream bool) {
	if !upstream {
		return
	}
	for i := range r.upstreamGradient.Data {
		r.upstreamGradient.Data[i] = 0
	}

	for y := 0; y < r.output.Height; y++ {
		for x := 0; x < r.output.Width; x++ {
			for z := 0; z < r.output.Depth; z++ {
				outputPoint := r.outputChoices[x][y][z]
				sourceX, sourceY := outputPoint[0], outputPoint[1]
				grad := r.downstreamGradient.Get(x, y, z)
				r.upstreamGradient.Set(sourceX, sourceY, z, grad)
			}
		}
	}
}

func (r *MaxPoolingLayer) GradientMagSquared() float64 {
	return 0
}

func (r *MaxPoolingLayer) StepGradient(f float64) {
}

func (r *MaxPoolingLayer) Alias() Layer {
	res := &MaxPoolingLayer{
		tensorLayer: tensorLayer{
			output: NewTensor3(r.output.Width, r.output.Height, r.output.Depth),
			upstreamGradient: NewTensor3(r.upstreamGradient.Width, r.upstreamGradient.Height,
				r.upstreamGradient.Depth),
		},
		xSpan:         r.xSpan,
		ySpan:         r.ySpan,
		outputChoices: make([][][][2]int, len(r.outputChoices)),
	}
	for i, outer := range r.outputChoices {
		newOuter := make([][][2]int, len(outer))
		for j, inner := range outer {
			newInner := make([][2]int, len(inner))
			newOuter[j] = newInner
		}
		res.outputChoices[i] = newOuter
	}
	return res
}

func (r *MaxPoolingLayer) Serialize() ([]byte, error) {
	s := serializedMaxPoolingLayer{
		InputWidth:  r.upstreamGradient.Width,
		InputHeight: r.upstreamGradient.Height,
		InputDepth:  r.upstreamGradient.Depth,

		OutputWidth:  r.output.Width,
		OutputHeight: r.output.Height,
		OutputDepth:  r.output.Depth,

		XSpan: r.xSpan,
		YSpan: r.ySpan,
	}
	return json.Marshal(&s)
}

func (r *MaxPoolingLayer) SerializerType() string {
	return serializerTypeMaxPoolingLayer
}

// maxInput computes the maxmimum input value
// in a given rectangular range.
func (r *MaxPoolingLayer) maxInput(x1, x2, y1, y2, z int) (value float64, bestX, bestY int) {
	value = math.Inf(-1)
	for x := x1; x <= x2; x++ {
		for y := y1; y <= y2; y++ {
			input := r.input.Get(x, y, z)
			if input > value {
				value = input
				bestX = x
				bestY = y
			}
		}
	}
	return
}

type serializedMaxPoolingLayer struct {
	InputWidth  int
	InputHeight int
	InputDepth  int

	OutputWidth  int
	OutputHeight int
	OutputDepth  int

	XSpan int
	YSpan int
}
