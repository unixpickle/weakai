package neuralnet

import (
	"encoding/json"
	"math/rand"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
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

	Filters []*Tensor3
	Biases  *autofunc.Variable

	// FilterVar must contain the data for all of the
	// filters in Filters, arranged one after the other.
	// The array behind the slice in FilterVar should
	// be re-used in Filters.
	FilterVar *autofunc.Variable `json:"-"`
}

func DeserializeConvLayer(data []byte) (*ConvLayer, error) {
	var c ConvLayer
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, err
	}

	filterSize := c.FilterWidth * c.FilterHeight * c.InputDepth
	weightCount := c.FilterCount * filterSize
	weightSlice := make(linalg.Vector, weightCount)
	for i, x := range c.Filters {
		subSlice := weightSlice[i*filterSize : (i+1)*filterSize]
		copy(subSlice, x.Data)
		x.Data = subSlice
	}
	c.FilterVar = &autofunc.Variable{Vector: weightSlice}

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
		filterSize := c.FilterWidth * c.FilterHeight * c.InputDepth
		weightCount := c.FilterCount * filterSize
		c.FilterVar = &autofunc.Variable{
			Vector: make(linalg.Vector, weightCount),
		}
		for i := 0; i < c.FilterCount; i++ {
			filter := &Tensor3{
				Width:  c.FilterWidth,
				Height: c.FilterHeight,
				Depth:  c.InputDepth,
				Data:   c.FilterVar.Vector[i*filterSize : (i+1)*filterSize],
			}
			c.Filters = append(c.Filters, filter)
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
// and filter variables.
func (c *ConvLayer) Parameters() []*autofunc.Variable {
	if c.Filters == nil || c.Biases == nil || c.FilterVar == nil {
		panic(uninitPanicMessage)
	}
	return []*autofunc.Variable{c.Biases, c.FilterVar}
}

// Apply computes convolutions on the input.
// The result is only valid as long as the ConvLayer
// that produced it (c, in this case) is not modified.
func (c *ConvLayer) Apply(in autofunc.Result) autofunc.Result {
	if c.Filters == nil || c.Biases == nil || c.FilterVar == nil {
		panic(uninitPanicMessage)
	}
	inMatrix, out := c.convolve(in.Output())
	return &convLayerResult{
		OutputTensor: out,
		InMatrix:     inMatrix,
		Input:        in,
		Layer:        c,
	}
}

// ApplyR is like Apply, but for autofunc.RResults.
func (c *ConvLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	if c.Filters == nil || c.Biases == nil || c.FilterVar == nil {
		panic(uninitPanicMessage)
	}
	_, outTensor := c.convolve(in.Output())
	return &convLayerRResult{
		OutputTensor:  outTensor,
		ROutputTensor: c.convolveR(v, in.Output(), in.ROutput()),
		Input:         in,
		Layer:         c,
		RV:            v,
	}
}

func (c *ConvLayer) Serialize() ([]byte, error) {
	return json.Marshal(c)
}

func (c *ConvLayer) SerializerType() string {
	return serializerTypeConvLayer
}

func (c *ConvLayer) convolve(input linalg.Vector) (in blas64.General, out *Tensor3) {
	inTensor := c.inputToTensor(input)
	outTensor := NewTensor3(c.OutputWidth(), c.OutputHeight(), c.OutputDepth())

	inMat := blas64.General{
		Rows:   outTensor.Width * outTensor.Height,
		Cols:   c.FilterWidth * c.FilterHeight * c.InputDepth,
		Stride: c.FilterWidth * c.FilterHeight * c.InputDepth,
		Data:   inTensor.ToCol(c.FilterWidth, c.FilterHeight, c.Stride),
	}
	filterMat := blas64.General{
		Rows:   c.FilterCount,
		Cols:   inMat.Cols,
		Stride: inMat.Stride,
		Data:   c.FilterVar.Vector,
	}
	outMat := blas64.General{
		Rows:   outTensor.Width * outTensor.Height,
		Cols:   outTensor.Depth,
		Stride: outTensor.Depth,
		Data:   outTensor.Data,
	}
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, inMat, filterMat, 0, outMat)

	biasVec := blas64.Vector{Inc: 1, Data: c.Biases.Vector}
	for i := 0; i < len(outTensor.Data); i += outMat.Cols {
		outRow := outTensor.Data[i : i+outMat.Cols]
		outVec := blas64.Vector{Inc: 1, Data: outRow}
		blas64.Axpy(len(outRow), 1, biasVec, outVec)
	}

	return inMat, outTensor
}

func (c *ConvLayer) convolveR(v autofunc.RVector, input, inputR linalg.Vector) *Tensor3 {
	inTensor := c.inputToTensor(input)
	inTensorR := c.inputToTensor(inputR)
	croppedInput := NewTensor3(c.FilterWidth, c.FilterHeight, c.InputDepth)
	croppedInputR := NewTensor3(c.FilterWidth, c.FilterHeight, c.InputDepth)
	outTensor := NewTensor3(c.OutputWidth(), c.OutputHeight(), c.OutputDepth())

	filtersR := c.filtersR(v)
	biasR := v[c.Biases]

	for y := 0; y < outTensor.Height; y++ {
		inputY := y * c.Stride
		for x := 0; x < outTensor.Width; x++ {
			inputX := x * c.Stride
			inTensor.Crop(inputX, inputY, croppedInput)
			inTensorR.Crop(inputX, inputY, croppedInputR)
			croppedVec := blas64.Vector{
				Inc:  1,
				Data: croppedInput.Data,
			}
			croppedVecR := blas64.Vector{
				Inc:  1,
				Data: croppedInputR.Data,
			}
			for z, filter := range c.Filters {
				filterVec := blas64.Vector{
					Inc:  1,
					Data: filter.Data,
				}
				convolution := blas64.Dot(len(filter.Data), filterVec, croppedVecR)
				if rfilter := filtersR[z]; rfilter != nil {
					filterVecR := blas64.Vector{
						Inc:  1,
						Data: rfilter.Data,
					}
					convolution += blas64.Dot(len(rfilter.Data), filterVecR, croppedVec)
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
		filters = make([]*Tensor3, len(c.Filters))
		return
	}

	bias = m[c.Biases]

	if filtersGrad := m[c.FilterVar]; filtersGrad != nil {
		for _, f := range c.Filters {
			gradData := filtersGrad[:f.Width*f.Height*f.Depth]
			filters = append(filters, c.filterToTensor(gradData))
			filtersGrad = filtersGrad[f.Width*f.Height*f.Depth:]
		}
	} else {
		filters = make([]*Tensor3, len(c.Filters))
	}

	return
}

func (c *ConvLayer) filtersR(v autofunc.RVector) []*Tensor3 {
	var filtersR []*Tensor3
	if filtersGrad := v[c.FilterVar]; filtersGrad != nil {
		for _, f := range c.Filters {
			gradData := filtersGrad[:f.Width*f.Height*f.Depth]
			filtersR = append(filtersR, c.filterToTensor(gradData))
			filtersGrad = filtersGrad[f.Width*f.Height*f.Depth:]
		}
	} else {
		filtersR = make([]*Tensor3, len(c.Filters))
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
	InMatrix     blas64.General
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
	return c.Layer.FilterVar.Constant(g)
}

func (c *convLayerResult) PropagateGradient(upstream linalg.Vector, grad autofunc.Gradient) {
	upstreamMat := blas64.General{
		Rows:   c.OutputTensor.Width * c.OutputTensor.Height,
		Cols:   c.OutputTensor.Depth,
		Stride: c.OutputTensor.Depth,
		Data:   upstream,
	}

	if biasGrad, ok := grad[c.Layer.Biases]; ok {
		biasGradVec := blas64.Vector{Inc: 1, Data: biasGrad}
		for i := 0; i < len(upstreamMat.Data); i += upstreamMat.Cols {
			row := blas64.Vector{
				Inc:  1,
				Data: upstreamMat.Data[i : i+upstreamMat.Cols],
			}
			blas64.Axpy(len(biasGrad), 1, row, biasGradVec)
		}
	}

	if !c.Input.Constant(grad) {
		inDeriv := c.InMatrix
		inDeriv.Data = make([]float64, len(c.InMatrix.Data))
		filterMat := blas64.General{
			Rows:   len(c.Layer.Filters),
			Cols:   c.Layer.FilterWidth * c.Layer.FilterHeight * c.Layer.InputDepth,
			Stride: c.Layer.FilterWidth * c.Layer.FilterHeight * c.Layer.InputDepth,
			Data:   c.Layer.FilterVar.Vector,
		}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, upstreamMat, filterMat, 0, inDeriv)
		flattened := NewTensor3Col(c.Layer.InputWidth, c.Layer.InputHeight,
			c.Layer.InputDepth, inDeriv.Data, c.Layer.FilterWidth,
			c.Layer.FilterHeight, c.Layer.Stride)
		c.Input.PropagateGradient(flattened.Data, grad)
	}

	if filterGrad, ok := grad[c.Layer.FilterVar]; ok {
		destMat := blas64.General{
			Rows:   len(c.Layer.Filters),
			Cols:   c.Layer.FilterWidth * c.Layer.FilterHeight * c.Layer.InputDepth,
			Stride: c.Layer.FilterWidth * c.Layer.FilterHeight * c.Layer.InputDepth,
			Data:   filterGrad,
		}
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, upstreamMat, c.InMatrix, 1, destMat)
	}
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

	if !c.Layer.FilterVar.Constant(g) {
		return false
	} else if _, ok := rg[c.Layer.FilterVar]; ok {
		return false
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
		inputGrad = NewTensor3(c.Layer.InputWidth, c.Layer.InputHeight, c.Layer.InputDepth)
		inputGradR = NewTensor3(c.Layer.InputWidth, c.Layer.InputHeight, c.Layer.InputDepth)
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
	}
}
