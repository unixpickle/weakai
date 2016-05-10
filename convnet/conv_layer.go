package main

type ConvLayer struct {
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

	InGradient  []float64
	OutGradient []float64
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
	c.OutGradient = make([]float64, c.InDepth*c.InWidth*c.InHeight)
}

// ComputeOut computes the output of this
// layer using the layer's inputs and its
// weights.
func (c *ConvLayer) ComputeOut() {
	featureRowLen := c.FeatureWidth * c.InDepth
	outputIdx := 0
	for y := 0; y <= c.InHeight-c.FeatureHeight; y += c.Stride {
		for x := 0; x <= c.InWidth-c.FeatureWidth; x += c.Stride {
			for _, featureWeights := range c.FeatureWeights {
				summer := kahan.Summer64()
				inRowStart := (c.InWidth*y + x) * c.InDepth
				weightIdx := 0
				for j := 0; j < c.FeatureHeight; j++ {
					for k := 0; k < featureRowLen; k++ {
						summer.Add(featureWeights[weightIdx] * c.InBuffer[inRowStart+k])
						weightIdx++
					}
					inRowStart += c.InWidth * c.InDepth
				}
				c.OutBuffer[outputIdx] = summer.Sum()
				outputIdx++
			}
		}
	}
}

// OutputDims returns the width and height
// (but not depth) of the output.
func (c *ConvLayer) OutputDims() (w, h int) {
	if c.FeatureWidth > c.InWidth || c.FeatureHeight > c.InHeight {
		return
	}
	w = 1 + (c.InWidth-c.FeatureWidth)/c.Stride
	h = 1 + (c.InHeight-c.FeatureHeight)/c.Stride
	return
}
