package convnet

import "math"

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

type MaxPoolingLayer struct {
	XSpan int
	YSpan int

	// Output is the output from this layer.
	// This is updated during forward-propagation.
	Output *Tensor3

	// UpstreamGradient is the gradient of the loss
	// function with respect to the input tensor.
	// This is updated during back-propagation.
	UpstreamGradient *Tensor3

	// Input is the input to this layer.
	// This should be set by an external party
	// before forward-propagation.
	Input *Tensor3

	// DownstreamGradient is the gradient of the
	// loss function with respect to the output
	// of this layer.
	// This should be set by an external party
	// before back-propagation.
	DownstreamGradient *Tensor3

	// outputChoices[x][y][z] is the pair {x1,y1}
	// indicating which value in the pool associated
	// with output (x,y,z) was chosen.
	// The pair {x1,y1,z} is the coordinate in
	// the input tensor which was output.
	outputChoices [][][][2]int
}

func NewMaxPoolingLayer(params *MaxPoolingParams) *MaxPoolingLayer {
	w := params.InputWidth - params.XSpan
	if (params.InputWidth % params.XSpan) != 0 {
		w++
	}
	h := params.InputHeight - params.YSpan
	if (params.InputHeight % params.YSpan) != 0 {
		h++
	}
	res := &MaxPoolingLayer{
		XSpan:            params.XSpan,
		YSpan:            params.YSpan,
		Output:           NewTensor3(w, h, params.InputDepth),
		UpstreamGradient: NewTensor3(params.InputWidth, params.InputHeight, params.InputDepth),
		outputChoices:    make([][][][2]int, w),
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

// Randomize does nothing, since this type of
// layer has no learnable values.
func (r *MaxPoolingLayer) Randomize() {
}

// PropagateForward performs forward-propagation.
func (r *MaxPoolingLayer) PropagateForward() {
	for y := 0; y < r.Output.Height; y++ {
		poolY := y * r.YSpan
		maxY := poolY + r.YSpan - 1
		if maxY >= r.Input.Height {
			maxY = r.Input.Height - 1
		}
		for x := 0; x < r.Output.Width; x++ {
			poolX := x * r.XSpan
			maxX := poolX + r.XSpan - 1
			if maxX >= r.Input.Width {
				maxX = r.Input.Width - 1
			}
			for z := 0; z < r.Output.Depth; z++ {
				output, bestX, bestY := r.maxInput(poolX, maxX, poolY, maxY, z)
				r.Output.Set(x, y, z, output)
				r.outputChoices[x][y][z] = [2]int{bestX, bestY}
			}
		}
	}
}

// PropagateBackward performs backward-propagation.
func (r *MaxPoolingLayer) PropagateBackward() {
	for y := 0; y < r.Output.Height; y++ {
		for x := 0; x < r.Output.Width; x++ {
			for z := 0; z < r.Output.Depth; z++ {
				outputPoint := r.outputChoices[x][y][z]
				sourceX, sourceY := outputPoint[0], outputPoint[1]
				grad := r.DownstreamGradient.Get(x, y, z)
				r.UpstreamGradient.Set(sourceX, sourceY, z, grad)
			}
		}
	}
}

// maxInput computes the maxmimum input value
// in a given rectangular range.
func (r *MaxPoolingLayer) maxInput(x1, x2, y1, y2, z int) (value float64, bestX, bestY int) {
	value = math.Inf(-1)
	for x := x1; x <= x2; x++ {
		for y := y1; y <= y2; y++ {
			input := r.Input.Get(x, y, z)
			if input > value {
				value = input
				bestX = x
				bestY = y
			}
		}
	}
	return
}
