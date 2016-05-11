package convnet

// ConvParams are parameters for constructing
// a ConvLayer.
type ConvParams struct {
	FilterCount  int
	FilterWidth  int
	FilterHeight int
	Stride       int

	InputWidth  int
	InputHeight int
	InputDepth  int

	Activation ActivationFunc
}

// ConvLayer is a convolutional layer for a
// neural network.
//It implements back- and forward-propagation.
type ConvLayer struct {
	// Activation is the activation function used
	// to turn each convolution into an output for
	// the next layer (e.g. a sigmoid function).
	Activation ActivationFunc

	// Stride is the amount that the filters should
	// "move" horizontally and vertically between
	// convolutions, i.e. the spacing in input space
	// corresponding to one-pixel difference in output
	// space.
	Stride int

	// Filters is an array of convolutional filters
	// used in the network.
	// Each filter is a tensor which corresponds to
	// the weights to be applied to the filter's
	// input values.
	Filters []*Tensor3

	// FilterGradients is structed exactly like
	// Filter, but after back-propagation, every
	// entry corresponds to the gradient with respect
	// to the entry in Filters.
	FilterGradients []*Tensor3

	// Biases is structured like Output, and stores the
	// bias value for each filter used to generate the
	// output.
	Biases *Tensor3

	// BiasGradients is structured like Biases, but
	// after back-propagation, every entry corresponds
	// to the partial of the cost function with respect
	// to the corresponding bias value.
	BiasGradients *Tensor3

	// Output is the ouput from this convolution layer.
	Output *Tensor3

	// UpstreamGradient is structured like Input.
	// Back-propagation sets values in UpstreamGradient
	// to specify the gradient of the cost function with
	// respect to the inputs to this layer.
	UpstreamGradient *Tensor3

	// Input is the input data to this layer.
	// This should be set by some external entity
	// before forward-propagation.
	Input *Tensor3

	// DownstreamGradient is the gradient of the
	// cost function with respect to the outputs
	// of this layer.
	// This should be set by some external entity
	// before back-propagation.
	DownstreamGradient *Tensor3

	// convolutions is the output from this convolution
	// layer, before applying the activation function.
	convolutions *Tensor3
}

func NewConvLayer(params *ConvParams) *ConvLayer {
	w := 1 + (params.InputWidth-params.FilterWidth)/params.Stride
	h := 1 + (params.InputHeight-params.FilterHeight)/params.Stride

	if w < 0 {
		w = 0
	}
	if h < 0 {
		h = 0
	}

	res := &ConvLayer{
		Activation: params.Activation,
		Stride:     params.Stride,

		Filters:         make([]*Tensor3, params.FilterCount),
		FilterGradients: make([]*Tensor3, params.FilterCount),
		Biases:          NewTensor3(w, h, params.FilterCount),

		Output:           NewTensor3(w, h, params.FilterCount),
		convolutions:     NewTensor3(w, h, params.FilterCount),
		UpstreamGradient: NewTensor3(params.InputWidth, params.InputHeight, params.InputDepth),
	}

	for i := 0; i < params.FilterCount; i++ {
		res.Filters[i] = NewTensor3(params.FilterWidth, params.FilterHeight, params.InputDepth)
		res.FilterGradients[i] = NewTensor3(params.FilterWidth, params.FilterHeight,
			params.InputDepth)
	}

	return res
}

// PropagateForward performs forward-propagation,
// computing the output convolutions and activations
// of this layer.
func (c *ConvLayer) PropagateForward() {
	for y := 0; y < c.Output.Height; y++ {
		inputY := y * c.Stride
		for x := 0; x < c.Output.Width; x++ {
			inputX := x * c.Stride
			for z, filter := range c.Filters {
				convolution := filter.Convolve(inputX, inputY, filter)
				convolution += c.Biases.Get(x, y, z)
				c.convolutions.Set(x, y, z, convolution)
				c.Output.Set(x, y, z, c.Activation.Eval(convolution))
			}
		}
	}
}

// PropagateBackward performs backward propagation.
// This must be called after ForwardPropagate.
func (c *ConvLayer) PropagateBackward() {
	for _, x := range c.FilterGradients {
		x.Reset()
	}
	c.UpstreamGradient.Reset()

	for y := 0; y < c.Output.Height; y++ {
		inputY := y * c.Stride
		for x := 0; x < c.Output.Width; x++ {
			inputX := x * c.Stride
			for z, filter := range c.Filters {
				sumPartial := c.DownstreamGradient.Get(x, y, z) *
					c.Activation.Deriv(c.convolutions.Get(x, y, z))
				c.FilterGradients[z].MulAdd(-inputX, -inputY, c.Input, sumPartial)
				c.UpstreamGradient.MulAdd(inputX, inputY, filter, sumPartial)
				c.BiasGradients.Set(x, y, z, sumPartial)
			}
		}
	}
}
