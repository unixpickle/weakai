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

// DeserializeConvLayer deserializes a ConvLayer.
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
	return c.Batch(in, 1)
}

// ApplyR is like Apply, but for autofunc.RResults.
func (c *ConvLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return c.BatchR(v, in, 1)
}

// Batch applies the layer to inputs in batch.
func (c *ConvLayer) Batch(in autofunc.Result, n int) autofunc.Result {
	if c.Filters == nil || c.Biases == nil || c.FilterVar == nil {
		panic(uninitPanicMessage)
	}
	outSize := c.OutputWidth() * c.OutputHeight() * c.OutputDepth()
	inSize := c.InputWidth * c.InputHeight * c.InputDepth
	if len(in.Output()) != n*inSize {
		panic("invalid input size")
	}
	res := &convLayerResult{
		OutputVec: make(linalg.Vector, outSize*n),
		Input:     in,
		N:         n,
		Layer:     c,
	}
	for i := 0; i < n; i++ {
		subIn := in.Output()[i*inSize : (i+1)*inSize]
		subOut := res.OutputVec[i*outSize : (i+1)*outSize]
		c.convolve(subIn, c.outputToTensor(subOut))
	}
	return res
}

// BatchR is like Batch, but for RResults.
func (c *ConvLayer) BatchR(rv autofunc.RVector, in autofunc.RResult,
	n int) autofunc.RResult {
	if c.Filters == nil || c.Biases == nil || c.FilterVar == nil {
		panic(uninitPanicMessage)
	}
	outSize := c.OutputWidth() * c.OutputHeight() * c.OutputDepth()
	inSize := c.InputWidth * c.InputHeight * c.InputDepth
	if len(in.Output()) != n*inSize {
		panic("invalid input size")
	}
	res := &convLayerRResult{
		OutputVec:  make(linalg.Vector, outSize*n),
		ROutputVec: make(linalg.Vector, outSize*n),
		Input:      in,
		FiltersR:   rv[c.FilterVar],
		N:          n,
		Layer:      c,
	}
	for i := 0; i < n; i++ {
		subIn := in.Output()[i*inSize : (i+1)*inSize]
		subOut := res.OutputVec[i*outSize : (i+1)*outSize]
		c.convolve(subIn, c.outputToTensor(subOut))

		subInR := in.ROutput()[i*inSize : (i+1)*inSize]
		subOutR := res.ROutputVec[i*outSize : (i+1)*outSize]
		c.convolveR(rv, subIn, subInR, c.outputToTensor(subOutR))
	}
	return res
}

// Serialize serializes the layer.
func (c *ConvLayer) Serialize() ([]byte, error) {
	return json.Marshal(c)
}

// SerializerType returns the unique ID used to serialize
// this layer with the serializer package.
func (c *ConvLayer) SerializerType() string {
	return serializerTypeConvLayer
}

func (c *ConvLayer) convolve(in linalg.Vector, out *Tensor3) {
	inMat := c.inputToMatrix(in)
	filterMat := blas64.General{
		Rows:   c.FilterCount,
		Cols:   inMat.Cols,
		Stride: inMat.Stride,
		Data:   c.FilterVar.Vector,
	}
	outMat := blas64.General{
		Rows:   out.Width * out.Height,
		Cols:   out.Depth,
		Stride: out.Depth,
		Data:   out.Data,
	}
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, inMat, filterMat, 0, outMat)

	biasVec := blas64.Vector{Inc: 1, Data: c.Biases.Vector}
	for i := 0; i < len(out.Data); i += outMat.Cols {
		outRow := out.Data[i : i+outMat.Cols]
		outVec := blas64.Vector{Inc: 1, Data: outRow}
		blas64.Axpy(len(outRow), 1, biasVec, outVec)
	}
}

func (c *ConvLayer) convolveR(v autofunc.RVector, in, inR linalg.Vector, out *Tensor3) {
	inMat := c.inputToMatrix(in)
	inMatR := c.inputToMatrix(inR)
	filterMat := blas64.General{
		Rows:   c.FilterCount,
		Cols:   inMat.Cols,
		Stride: inMat.Stride,
		Data:   c.FilterVar.Vector,
	}
	outMat := blas64.General{
		Rows:   out.Width * out.Height,
		Cols:   out.Depth,
		Stride: out.Depth,
		Data:   out.Data,
	}
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, inMatR, filterMat, 0, outMat)
	if filterRV, ok := v[c.FilterVar]; ok {
		filterMatR := blas64.General{
			Rows:   c.FilterCount,
			Cols:   inMat.Cols,
			Stride: inMat.Stride,
			Data:   filterRV,
		}
		blas64.Gemm(blas.NoTrans, blas.Trans, 1, inMat, filterMatR, 1, outMat)
	}

	if biasRV, ok := v[c.Biases]; ok {
		biasVec := blas64.Vector{Inc: 1, Data: biasRV}
		for i := 0; i < len(out.Data); i += outMat.Cols {
			outRow := out.Data[i : i+outMat.Cols]
			outVec := blas64.Vector{Inc: 1, Data: outRow}
			blas64.Axpy(len(outRow), 1, biasVec, outVec)
		}
	}
}

func (c *ConvLayer) inputToTensor(in linalg.Vector) *Tensor3 {
	return &Tensor3{
		Width:  c.InputWidth,
		Height: c.InputHeight,
		Depth:  c.InputDepth,
		Data:   in,
	}
}

func (c *ConvLayer) inputToMatrix(in linalg.Vector) blas64.General {
	inTensor := c.inputToTensor(in)
	return blas64.General{
		Rows:   c.OutputWidth() * c.OutputHeight(),
		Cols:   c.FilterWidth * c.FilterHeight * c.InputDepth,
		Stride: c.FilterWidth * c.FilterHeight * c.InputDepth,
		Data:   inTensor.ToCol(c.FilterWidth, c.FilterHeight, c.Stride),
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

type convLayerResult struct {
	OutputVec linalg.Vector
	Input     autofunc.Result
	N         int
	Layer     *ConvLayer
}

func (c *convLayerResult) Output() linalg.Vector {
	return c.OutputVec
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
	c.propagateBiases(upstream, grad)

	var inputDownstream linalg.Vector
	if !c.Input.Constant(grad) {
		inputDownstream = make(linalg.Vector, len(c.Input.Output()))
	}

	subUpstreamSize := len(upstream) / c.N
	subDownstreamSize := len(c.Input.Output()) / c.N
	for i := 0; i < c.N; i++ {
		subUpstream := upstream[i*subUpstreamSize : (i+1)*subUpstreamSize]
		var subDownstream linalg.Vector
		if inputDownstream != nil {
			subDownstream = inputDownstream[i*subDownstreamSize : (i+1)*subDownstreamSize]
		}
		subInput := c.Input.Output()[i*subDownstreamSize : (i+1)*subDownstreamSize]
		c.propagateSingle(subInput, subUpstream, subDownstream, grad)
	}

	if !c.Input.Constant(grad) {
		c.Input.PropagateGradient(inputDownstream, grad)
	}
}

func (c *convLayerResult) propagateBiases(upstream linalg.Vector, grad autofunc.Gradient) {
	if biasGrad, ok := grad[c.Layer.Biases]; ok {
		biasGradVec := blas64.Vector{Inc: 1, Data: biasGrad}
		for i := 0; i < len(upstream); i += c.Layer.OutputDepth() {
			row := blas64.Vector{
				Inc:  1,
				Data: upstream[i : i+c.Layer.OutputDepth()],
			}
			blas64.Axpy(len(biasGrad), 1, row, biasGradVec)
		}
	}
}

func (c *convLayerResult) propagateSingle(input, upstream, downstream linalg.Vector,
	grad autofunc.Gradient) {
	upstreamMat := blas64.General{
		Rows:   c.Layer.OutputWidth() * c.Layer.OutputHeight(),
		Cols:   c.Layer.OutputDepth(),
		Stride: c.Layer.OutputDepth(),
		Data:   upstream,
	}

	inMatrix := c.Layer.inputToMatrix(input)

	if filterGrad, ok := grad[c.Layer.FilterVar]; ok {
		destMat := blas64.General{
			Rows:   len(c.Layer.Filters),
			Cols:   c.Layer.FilterWidth * c.Layer.FilterHeight * c.Layer.InputDepth,
			Stride: c.Layer.FilterWidth * c.Layer.FilterHeight * c.Layer.InputDepth,
			Data:   filterGrad,
		}
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, upstreamMat, inMatrix, 1, destMat)
	}

	if downstream != nil {
		inDeriv := inMatrix
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
		copy(downstream, flattened.Data)
	}
}

type convLayerRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      autofunc.RResult
	FiltersR   linalg.Vector
	N          int
	Layer      *ConvLayer
}

func (c *convLayerRResult) Output() linalg.Vector {
	return c.OutputVec
}

func (c *convLayerRResult) ROutput() linalg.Vector {
	return c.ROutputVec
}

func (c *convLayerRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
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

	if !c.Input.Constant(rg, g) {
		return false
	}
	return true
}

func (c *convLayerRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad autofunc.RGradient, grad autofunc.Gradient) {
	if grad == nil {
		grad = autofunc.Gradient{}
	}
	c.propagateBiases(upstream, upstreamR, rgrad, grad)

	var inputDownstream, inputDownstreamR linalg.Vector
	if !c.Input.Constant(rgrad, grad) {
		inputDownstream = make(linalg.Vector, len(c.Input.Output()))
		inputDownstreamR = make(linalg.Vector, len(c.Input.Output()))
	}

	subUpstreamSize := len(upstream) / c.N
	subDownstreamSize := len(c.Input.Output()) / c.N
	for i := 0; i < c.N; i++ {
		subUpstream := upstream[i*subUpstreamSize : (i+1)*subUpstreamSize]
		subUpstreamR := upstreamR[i*subUpstreamSize : (i+1)*subUpstreamSize]
		var subDownstream, subDownstreamR linalg.Vector
		if inputDownstream != nil {
			subDownstream = inputDownstream[i*subDownstreamSize : (i+1)*subDownstreamSize]
			subDownstreamR = inputDownstreamR[i*subDownstreamSize : (i+1)*subDownstreamSize]
		}
		subInput := c.Input.Output()[i*subDownstreamSize : (i+1)*subDownstreamSize]
		subInputR := c.Input.ROutput()[i*subDownstreamSize : (i+1)*subDownstreamSize]
		c.propagateSingle(subInput, subInputR, subUpstream, subUpstreamR,
			subDownstream, subDownstreamR, rgrad, grad)
	}

	if !c.Input.Constant(rgrad, grad) {
		c.Input.PropagateRGradient(inputDownstream, inputDownstreamR, rgrad, grad)
	}
}

func (c *convLayerRResult) propagateBiases(upstream, upstreamR linalg.Vector,
	rgrad autofunc.RGradient, grad autofunc.Gradient) {
	if biasGrad, ok := grad[c.Layer.Biases]; ok {
		biasGradVec := blas64.Vector{Inc: 1, Data: biasGrad}
		for i := 0; i < len(upstream); i += c.Layer.OutputDepth() {
			row := blas64.Vector{
				Inc:  1,
				Data: upstream[i : i+c.Layer.OutputDepth()],
			}
			blas64.Axpy(len(biasGrad), 1, row, biasGradVec)
		}
	}
	if biasRGrad, ok := rgrad[c.Layer.Biases]; ok {
		biasRGradVec := blas64.Vector{Inc: 1, Data: biasRGrad}
		for i := 0; i < len(upstream); i += c.Layer.OutputDepth() {
			row := blas64.Vector{
				Inc:  1,
				Data: upstreamR[i : i+c.Layer.OutputDepth()],
			}
			blas64.Axpy(len(biasRGrad), 1, row, biasRGradVec)
		}
	}
}

func (c *convLayerRResult) propagateSingle(input, inputR, upstream, upstreamR, downstream,
	downstreamR linalg.Vector, rgrad autofunc.RGradient, grad autofunc.Gradient) {
	upstreamMat := blas64.General{
		Rows:   c.Layer.OutputWidth() * c.Layer.OutputHeight(),
		Cols:   c.Layer.OutputDepth(),
		Stride: c.Layer.OutputDepth(),
		Data:   upstream,
	}
	upstreamMatR := blas64.General{
		Rows:   c.Layer.OutputWidth() * c.Layer.OutputHeight(),
		Cols:   c.Layer.OutputDepth(),
		Stride: c.Layer.OutputDepth(),
		Data:   upstreamR,
	}

	if downstream != nil {
		inDeriv := c.Layer.inputToMatrix(input)
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
		copy(downstream, flattened.Data)

		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, upstreamMatR, filterMat, 0, inDeriv)
		if c.FiltersR != nil {
			filterMat.Data = c.FiltersR
			blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, upstreamMat, filterMat, 1, inDeriv)
		}
		flattened = NewTensor3Col(c.Layer.InputWidth, c.Layer.InputHeight,
			c.Layer.InputDepth, inDeriv.Data, c.Layer.FilterWidth,
			c.Layer.FilterHeight, c.Layer.Stride)
		copy(downstreamR, flattened.Data)
	}

	filterGrad, hasFilterGrad := grad[c.Layer.FilterVar]
	filterRGrad, hasFilterRGrad := rgrad[c.Layer.FilterVar]

	var inMatrix blas64.General
	if hasFilterGrad || hasFilterRGrad {
		inMatrix = c.Layer.inputToMatrix(input)
	}

	if hasFilterGrad {
		destMat := blas64.General{
			Rows:   len(c.Layer.Filters),
			Cols:   c.Layer.FilterWidth * c.Layer.FilterHeight * c.Layer.InputDepth,
			Stride: c.Layer.FilterWidth * c.Layer.FilterHeight * c.Layer.InputDepth,
			Data:   filterGrad,
		}
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, upstreamMat, inMatrix, 1, destMat)
	}

	if hasFilterRGrad {
		inMatrixR := c.Layer.inputToMatrix(inputR)
		destMat := blas64.General{
			Rows:   len(c.Layer.Filters),
			Cols:   c.Layer.FilterWidth * c.Layer.FilterHeight * c.Layer.InputDepth,
			Stride: c.Layer.FilterWidth * c.Layer.FilterHeight * c.Layer.InputDepth,
			Data:   filterRGrad,
		}
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, upstreamMatR, inMatrix, 1, destMat)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, upstreamMat, inMatrixR, 1, destMat)
	}
}
