package neuralnet

import (
	"encoding/json"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// ConvLayer is a convolutional layer for
// a neural network.
type ConvLayer struct {
	FilterCount  int
	FilterWidth  int
	FilterHeight int
	Stride       int

	InputWidth  int
	InputHeight int
	InputDepth  int

	Cache *autofunc.VectorCache

	Filters    []*Tensor3
	FilterVars []*autofunc.Variable `json:"-"`
	Biases     *autofunc.Variable
}

func DeserializeConvLayer(data []byte) (*ConvLayer, error) {
	var c ConvLayer
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, err
	}

	for _, x := range c.Filters {
		v := &autofunc.Variable{Vector: x.Data}
		c.FilterVars = append(c.FilterVars, v)
	}

	return &c, nil
}

// OutputWidth computes the width of the output tensor.
func (c *ConvLayer) OutputWidth() int {
	w := 1 + (c.InputWidth-c.FilterWidth)/c.Stride
	if w < 0 {
		return 0
	}
	return w
}

// OutputHeight computes the height of the output tensor.
func (c *ConvLayer) OutputHeight() int {
	h := 1 + (c.InputHeight-c.FilterHeight)/c.Stride
	if h < 0 {
		return 0
	}
	return h
}

// OutputDepth returns the depth of the output tensor.
func (c *ConvLayer) OutputDepth() int {
	return c.FilterCount
}

// Randomize randomly initializes the layer's
// filters and biases.
// This will allocate c.Filters, c.Biases,
// c.FilterVars, and c.BiasVars if needed.
func (c *ConvLayer) Randomize() {
	if c.Filters == nil {
		for i := 0; i < c.FilterCount; i++ {
			filter := NewTensor3(c.FilterWidth, c.FilterHeight, c.InputDepth)
			filterVar := &autofunc.Variable{Vector: linalg.Vector(filter.Data)}
			c.Filters = append(c.Filters, filter)
			c.FilterVars = append(c.FilterVars, filterVar)
		}
	}
	if c.Biases == nil {
		biasVec := make(linalg.Vector, c.FilterCount)
		c.Biases = &autofunc.Variable{Vector: biasVec}
	}
	for i, filter := range c.Filters {
		filter.Randomize()
		c.Biases.Vector[i] = (rand.Float64() * 2) - 1
	}
}

// Parameters returns a slice containing the bias
// variable and all the filter variables.
func (c *ConvLayer) Parameters() []*autofunc.Variable {
	res := make([]*autofunc.Variable, len(c.FilterVars)+1)
	res[0] = c.Biases
	copy(res[1:], c.FilterVars)
	return res
}

// Apply computes convolutions on the input.
// The result is only valid as long as the ConvLayer
// that produced it (c, in this case) is not modified.
func (c *ConvLayer) Apply(in autofunc.Result) autofunc.Result {
	return &convLayerResult{
		OutputTensor: c.convolve(in.Output()),
		Input:        in,
		Layer:        c,
	}
}

// ApplyR is like Apply, but for autofunc.RResults.
func (c *ConvLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return &convLayerRResult{
		OutputTensor:  c.convolve(in.Output()),
		ROutputTensor: c.convolveR(v, in.Output(), in.ROutput()),
		Input:         in,
		Layer:         c,
		RV:            v,
	}
}

func (c *ConvLayer) SetCache(cache *autofunc.VectorCache) {
	c.Cache = cache
}

func (c *ConvLayer) Serialize() ([]byte, error) {
	return json.Marshal(c)
}

func (c *ConvLayer) SerializerType() string {
	return serializerTypeConvLayer
}

func (c *ConvLayer) convolve(input linalg.Vector) *Tensor3 {
	inTensor := c.inputToTensor(input)
	outTensor := NewTensor3Cache(c.Cache, c.OutputWidth(), c.OutputHeight(), c.OutputDepth())

	for y := 0; y < outTensor.Height; y++ {
		inputY := y * c.Stride
		for x := 0; x < outTensor.Width; x++ {
			inputX := x * c.Stride
			for z, filter := range c.Filters {
				convolution := filter.Convolve(inputX, inputY, inTensor)
				convolution += c.Biases.Vector[z]
				outTensor.Set(x, y, z, convolution)
			}
		}
	}

	return outTensor
}

func (c *ConvLayer) convolveR(v autofunc.RVector, input, inputR linalg.Vector) *Tensor3 {
	inTensor := c.inputToTensor(input)
	inTensorR := c.inputToTensor(inputR)
	outTensor := NewTensor3Cache(c.Cache, c.OutputWidth(), c.OutputHeight(), c.OutputDepth())

	filtersR := c.filtersR(v)
	biasR := v[c.Biases]

	for y := 0; y < outTensor.Height; y++ {
		inputY := y * c.Stride
		for x := 0; x < outTensor.Width; x++ {
			inputX := x * c.Stride
			for z, filter := range c.Filters {
				convolution := filter.Convolve(inputX, inputY, inTensorR)
				if rfilter := filtersR[z]; rfilter != nil {
					convolution += rfilter.Convolve(inputX, inputY, inTensor)
				}
				if biasR != nil {
					convolution += biasR[z]
				}
				outTensor.Set(x, y, z, convolution)
			}
		}
	}

	return outTensor
}

func (c *ConvLayer) gradsFromMap(m map[*autofunc.Variable]linalg.Vector) (bias linalg.Vector,
	filters []*Tensor3) {
	if m == nil {
		for _ = range c.FilterVars {
			filters = append(filters, nil)
		}
		return
	}

	bias = m[c.Biases]

	for _, v := range c.FilterVars {
		if gradVec := m[v]; gradVec == nil {
			filters = append(filters, nil)
		} else {
			filters = append(filters, c.filterToTensor(gradVec))
		}
	}

	return
}

func (c *ConvLayer) filtersR(v autofunc.RVector) []*Tensor3 {
	var filtersR []*Tensor3
	for _, filterVar := range c.FilterVars {
		data := v[filterVar]
		if data == nil {
			filtersR = append(filtersR, nil)
		} else {
			filtersR = append(filtersR, c.filterToTensor(data))
		}
	}
	return filtersR
}

func (c *ConvLayer) inputToTensor(in linalg.Vector) *Tensor3 {
	return &Tensor3{
		Width:  c.InputWidth,
		Height: c.InputHeight,
		Depth:  c.InputDepth,
		Data:   in,
	}
}

func (c *ConvLayer) outputToTensor(out linalg.Vector) *Tensor3 {
	return &Tensor3{
		Width:  c.OutputWidth(),
		Height: c.OutputHeight(),
		Depth:  c.OutputDepth(),
		Data:   out,
	}
}

func (c *ConvLayer) filterToTensor(filter linalg.Vector) *Tensor3 {
	return &Tensor3{
		Width:  c.FilterWidth,
		Height: c.FilterHeight,
		Depth:  c.InputDepth,
		Data:   filter,
	}
}

type convLayerResult struct {
	OutputTensor *Tensor3
	Input        autofunc.Result
	Layer        *ConvLayer
}

func (c *convLayerResult) Output() linalg.Vector {
	return c.OutputTensor.Data
}

func (c *convLayerResult) Constant(g autofunc.Gradient) bool {
	if !c.Layer.Biases.Constant(g) {
		return false
	}
	if !c.Input.Constant(g) {
		return false
	}
	for _, x := range c.Layer.FilterVars {
		if !x.Constant(g) {
			return false
		}
	}
	return true
}

func (c *convLayerResult) PropagateGradient(upstream linalg.Vector, grad autofunc.Gradient) {
	inputTensor := c.Layer.inputToTensor(c.Input.Output())
	downstreamTensor := c.Layer.outputToTensor(upstream)

	biasGrad, filterGrads := c.Layer.gradsFromMap(grad)

	var inputGrad *Tensor3
	if !c.Input.Constant(grad) {
		inputGrad = NewTensor3Cache(c.Layer.Cache, c.Layer.InputWidth, c.Layer.InputHeight,
			c.Layer.InputDepth)
	}

	for y := 0; y < c.OutputTensor.Height; y++ {
		inputY := y * c.Layer.Stride
		for x := 0; x < c.OutputTensor.Width; x++ {
			inputX := x * c.Layer.Stride
			for z, filter := range c.Layer.Filters {
				sumPartial := downstreamTensor.Get(x, y, z)
				if filterGrad := filterGrads[z]; filterGrad != nil {
					filterGrad.MulAdd(-inputX, -inputY, inputTensor, sumPartial)
				}
				if biasGrad != nil {
					biasGrad[z] += sumPartial
				}
				if inputGrad != nil {
					inputGrad.MulAdd(inputX, inputY, filter, sumPartial)
				}
			}
		}
	}

	if inputGrad != nil {
		c.Input.PropagateGradient(inputGrad.Data, grad)
		c.Layer.Cache.Free(inputGrad.Data)
	}
}

func (c *convLayerResult) Release() {
	c.Layer.Cache.Free(c.OutputTensor.Data)
	c.OutputTensor.Data = nil
	c.Input.Release()
}

type convLayerRResult struct {
	OutputTensor  *Tensor3
	ROutputTensor *Tensor3
	Input         autofunc.RResult
	Layer         *ConvLayer
	RV            autofunc.RVector
}

func (c *convLayerRResult) Output() linalg.Vector {
	return c.OutputTensor.Data
}

func (c *convLayerRResult) ROutput() linalg.Vector {
	return c.ROutputTensor.Data
}

func (c *convLayerRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	if !c.Input.Constant(rg, g) {
		return false
	}

	if !c.Layer.Biases.Constant(g) {
		return false
	} else if _, ok := rg[c.Layer.Biases]; ok {
		return false
	}

	for _, x := range c.Layer.FilterVars {
		if !x.Constant(g) {
			return false
		} else if _, ok := rg[x]; ok {
			return false
		}
	}

	return true
}

func (c *convLayerRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad autofunc.RGradient, grad autofunc.Gradient) {
	inputTensor := c.Layer.inputToTensor(c.Input.Output())
	inputTensorR := c.Layer.inputToTensor(c.Input.ROutput())
	downstreamTensor := c.Layer.outputToTensor(upstream)
	downstreamTensorR := c.Layer.outputToTensor(upstreamR)

	biasGrad, filterGrads := c.Layer.gradsFromMap(grad)
	biasGradR, filterGradsR := c.Layer.gradsFromMap(rgrad)

	var inputGrad *Tensor3
	var inputGradR *Tensor3

	if !c.Input.Constant(rgrad, grad) {
		inputGrad = NewTensor3Cache(c.Layer.Cache, c.Layer.InputWidth, c.Layer.InputHeight,
			c.Layer.InputDepth)
		inputGradR = NewTensor3Cache(c.Layer.Cache, c.Layer.InputWidth, c.Layer.InputHeight,
			c.Layer.InputDepth)
	}

	filtersR := c.Layer.filtersR(c.RV)

	for y := 0; y < c.OutputTensor.Height; y++ {
		inputY := y * c.Layer.Stride
		for x := 0; x < c.OutputTensor.Width; x++ {
			inputX := x * c.Layer.Stride
			for z, filter := range c.Layer.Filters {
				sumPartial := downstreamTensor.Get(x, y, z)
				sumPartialR := downstreamTensorR.Get(x, y, z)
				if filterGrad := filterGrads[z]; filterGrad != nil {
					filterGrad.MulAdd(-inputX, -inputY, inputTensor, sumPartial)
				}
				if filterGradR := filterGradsR[z]; filterGradR != nil {
					filterGradR.MulAdd(-inputX, -inputY, inputTensor, sumPartialR)
					filterGradR.MulAdd(-inputX, -inputY, inputTensorR, sumPartial)
				}
				if biasGrad != nil {
					biasGrad[z] += sumPartial
				}
				if biasGradR != nil {
					biasGradR[z] += sumPartialR
				}
				if inputGrad != nil {
					inputGrad.MulAdd(inputX, inputY, filter, sumPartial)
					inputGradR.MulAdd(inputX, inputY, filter, sumPartialR)
					if rfilter := filtersR[z]; rfilter != nil {
						inputGradR.MulAdd(inputX, inputY, rfilter, sumPartial)
					}
				}
			}
		}
	}

	if inputGrad != nil {
		c.Input.PropagateRGradient(inputGrad.Data, inputGradR.Data, rgrad, grad)
		c.Layer.Cache.Free(inputGrad.Data)
		c.Layer.Cache.Free(inputGradR.Data)
	}
}

func (c *convLayerRResult) Release() {
	c.Layer.Cache.Free(c.OutputTensor.Data)
	c.Layer.Cache.Free(c.ROutputTensor.Data)
	c.OutputTensor.Data = nil
	c.ROutputTensor.Data = nil
	c.Input.Release()
}
