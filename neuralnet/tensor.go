package neuralnet

import (
	"math"
	"math/rand"
)

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
	for y := 0; y < t.Height; y++ {
		for x := 0; x < t.Width; x++ {
			for z := 0; z < t.Depth; z++ {
				tVal := t.Get(x, y, z)
				t1Val := t1.Get(x+x1, y+y1, z)
				sum += tVal * t1Val
			}
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

	for y := 0; y < yCount; y++ {
		for x := 0; x < xCount; x++ {
			for z := 0; z < t.Depth; z++ {
				val1 := t.Get(x+targetStartX, y+targetStartY, z)
				val2 := t1.Get(x+sourceStartX, y+sourceStartY, z)
				t.Set(x+targetStartX, y+targetStartY, z, val1+(val2*s))
			}
		}
	}
}
