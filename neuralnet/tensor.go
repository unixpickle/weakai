package neuralnet

import (
	"math"
	"math/rand"

	"github.com/gonum/blas/blas64"
)

// minOptimizeTensorRowSize is the minimum row-size in
// a tensor required to have that tensor be effectively
// optimized with a BLAS package.
//
// This was chosen based on experiments on an Intel
// Core i7 using Go 1.7 and a CPU-based native BLAS.
// Results may vary.
//
// To test optimizations properly, it may be necessary
// to reduce this to 0.
const minOptimizeTensorRowSize = 16

// Tensor3 represents a 3D tensor, with
// values along an x, y, and z axis.
type Tensor3 struct {
	Width  int
	Height int
	Depth  int
	Data   []float64
}

// NewTensor3 creates an all-zero tensor of the
// given dimensions.
func NewTensor3(width, height, depth int) *Tensor3 {
	return &Tensor3{
		Width:  width,
		Height: height,
		Depth:  depth,
		Data:   make([]float64, width*height*depth),
	}
}

// Reset sets all entries to zero.
func (t *Tensor3) Reset() {
	for i := range t.Data {
		t.Data[i] = 0
	}
}

// Randomize sets all the entries of this
// tensor to random numbers in such a way
// that the standard deviation of the sum
// of all the entries is 1.
func (t *Tensor3) Randomize() {
	coeff := math.Sqrt(3.0 / float64(len(t.Data)))
	for i := range t.Data {
		t.Data[i] = coeff * ((rand.Float64() * 2) - 1)
	}
}

// Get returns the element of the tensor at
// the given indices.
func (t *Tensor3) Get(x, y, z int) float64 {
	return t.Data[(x+y*t.Width)*t.Depth+z]
}

// Set changes the element of the tensor at
// the given indices.
func (t *Tensor3) Set(x, y, z int, val float64) {
	t.Data[(x+y*t.Width)*t.Depth+z] = val
}

// Convolve convolves the filter t
// with a tensor t1, starting at x1,
// y1 in t1.
//
// Both tensors must have the same depth.
func (t *Tensor3) Convolve(x1, y1 int, t1 *Tensor3) float64 {
	if t.Depth != t1.Depth {
		panic("depths must match")
	}

	var sum float64
	if rowSize := t.Width * t.Depth; rowSize < minOptimizeTensorRowSize {
		for y := 0; y < t.Height; y++ {
			for x := 0; x < t.Width; x++ {
				for z := 0; z < t.Depth; z++ {
					tVal := t.Get(x, y, z)
					t1Val := t1.Get(x+x1, y+y1, z)
					sum += tVal * t1Val
				}
			}
		}
	} else {
		tVec := t.Data
		for y := 0; y < t.Height; y++ {
			t1Vec := t1.Data[(t1.Width*(y+y1)+x1)*t1.Depth:]
			v := blas64.Vector{Inc: 1, Data: tVec}
			v1 := blas64.Vector{Inc: 1, Data: t1Vec}
			sum += blas64.Dot(rowSize, v, v1)
			tVec = tVec[rowSize:]
		}
	}
	return sum
}

// MulAdd adds the tensor t1, shifted
// by x1 and y1, and scaled by s, to t.
// It modifies t but leaves t1 alone.
//
// For instance, if x1=1 and y1=0, then
// the first column of t is not affected,
// and the first column of t1 is added to
// the second column of t.
//
// Both tensors must have the same depth.
func (t *Tensor3) MulAdd(x, y int, t1 *Tensor3, s float64) {
	if t.Depth != t1.Depth {
		panic("depths must match")
	}

	var sourceStartX, targetStartX int
	if x > 0 {
		targetStartX = x
	} else {
		sourceStartX = -x
	}

	var sourceStartY, targetStartY int
	if y > 0 {
		targetStartY = y
	} else {
		sourceStartY = -y
	}

	yCount := t.Height - targetStartY
	xCount := t.Width - targetStartX

	if sourceLimit := t1.Height - sourceStartY; sourceLimit < yCount {
		yCount = sourceLimit
	}
	if sourceLimit := t1.Width - sourceStartX; sourceLimit < xCount {
		xCount = sourceLimit
	}

	if rowSize := xCount * t.Depth; rowSize < minOptimizeTensorRowSize {
		for y := 0; y < yCount; y++ {
			for x := 0; x < xCount; x++ {
				for z := 0; z < t.Depth; z++ {
					val1 := t.Get(x+targetStartX, y+targetStartY, z)
					val2 := t1.Get(x+sourceStartX, y+sourceStartY, z)
					t.Set(x+targetStartX, y+targetStartY, z, val1+(val2*s))
				}
			}
		}
	} else {
		for y := 0; y < yCount; y++ {
			target := t.Data[((y+targetStartY)*t.Width+targetStartX)*t.Depth:]
			source := t1.Data[((y+sourceStartY)*t1.Width+sourceStartX)*t1.Depth:]
			targetVec := blas64.Vector{Inc: 1, Data: target}
			sourceVec := blas64.Vector{Inc: 1, Data: source}
			blas64.Axpy(rowSize, s, sourceVec, targetVec)
		}
	}
}
