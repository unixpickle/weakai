package convnet

type ConvLayer struct {
	Activation ActivationFunc

	InDepth  int
	InWidth  int
	InHeight int

	FeatureCount  int
	FeatureWidth  int
	FeatureHeight int
	Stride        int

	FeatureWeights [][]float64

	InBuffer  []float64
	OutBuffer []float64

	OutConvolutions []float64

	WeightGradient [][]float64
	InGradient     []float64
	OutGradient    []float64
}

// MakeSlices allocates the output buffer,
// output gradient, and feature weights.
// It does not allocate input slices, since
// these will come from other layers.
func (c *ConvLayer) MakeSlices() {
	c.FeatureWeights = make([][]float64, c.FeatureCount)
	for i := range c.FeatureWeights {
		c.FeatureWeights[i] = make([]float64, c.FeatureWidth*c.FeatureHeight*c.InDepth)
	}
	outWidth, outHeight := c.OutputDims()
	c.OutBuffer = make([]float64, outWidth*outHeight*c.FeatureCount)
	c.OutConvolutions = make([]float64, len(c.OutBuffer))
	c.OutGradient = make([]float64, c.InDepth*c.InWidth*c.InHeight)
}

// ComputeOut computes the output of this
// layer using the layer's inputs and its
// weights.
func (c *ConvLayer) ComputeOut() {
	outputIdx := 0
	for y := 0; y <= c.InHeight-c.FeatureHeight; y += c.Stride {
		for x := 0; x <= c.InWidth-c.FeatureWidth; x += c.Stride {
			for featureIdx := range c.FeatureWeights {
				convolution := c.convolve(x, y, featureIdx)
				c.OutConvolutions[outputIdx] = convolution
				c.OutBuffer[outputIdx] = c.Activation(convolution)
				outputIdx++
			}
		}
	}
}

// ComputeGradients computes the gradient of
// the loss function for each of the input
// values to this layer, and the gradient of
// the loss function for each of the weights
// of this layer.
//
// This requires that the layer's outputs have
// been computed and that its output gradients
// are correctly set.
func (c *ConvLayer) ComputeGradients() {
	gradientSums := make([][]*kahan.Summer64, len(c.WeightGradient))
	for i, x := range c.WeightGradient {
		gradientSums[i] = make([]*kahan.Summer64, len(x))
		for j := range x {
			gradientSums[i][j] = kahan.NewSummer64()
		}
	}

	// TODO: create kahan summers for each of the inputs,
	// since these inputs may contribute to a number of
	// different filter instances.

	outputIdx := 0
	for y := 0; y <= c.InHeight-c.FeatureHeight; y += c.Stride {
		for x := 0; x <= c.InWidth-c.FeatureWidth; x += c.Stride {
			for featureIdx := range c.FeatureWeights {
				activDeriv := c.Activation.Deriv(c.OutConvolutions[outputIdx])
				activDeriv *= c.OutGradient[outputIdx]
				// TODO: compute the gradient with respect to each
				// input, and with respect to each weight.
				outputIdx++
			}
		}
	}

	for i, x := range gradientSums {
		for j, s := range x {
			c.WeightGradient[i][j] = s.Sum()
		}
	}
}

// OutputDims returns the width and height
// of the output.
func (c *ConvLayer) OutputDims() (w, h int) {
	if c.FeatureWidth > c.InWidth || c.FeatureHeight > c.InHeight {
		return
	}
	w = 1 + (c.InWidth-c.FeatureWidth)/c.Stride
	h = 1 + (c.InHeight-c.FeatureHeight)/c.Stride
	return
}

func (c *ConvLayer) convolve(x, y, featureIdx int) float64 {
	summer := kahan.Summer64()

	kernel := c.FeatureWeights[featureIdx]
	floatsPerRow := c.FeatureWidth * c.InDepth

	weightIdx := 0
	inputRowStart := (c.InWidth*y + x) * c.InDepth
	for j := 0; j < c.FeatureHeight; j++ {
		for k := 0; k < floatsPerRow; k++ {
			summer.Add(kernel[weightIdx] * c.InBuffer[inputRowStart+k])
			weightIdx++
		}
		inputRowStart += c.InWidth * c.InDepth
	}

	return summer.Sum()
}
