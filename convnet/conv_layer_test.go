package convnet

import (
	"math"
	"testing"
)

func TestConvDimensions(t *testing.T) {
	paramsList := []*ConvParams{
		{1, 3, 3, 1, 9, 9, 1, nil},
		{5, 4, 7, 2, 17, 56, 18, nil},
	}

	outputDims := [][3]int{
		{7, 7, 1},
		{7, 25, 5},
	}

	for i, params := range paramsList {
		layer := NewConvLayer(params)
		expOutDims := outputDims[i]
		if layer.output.Width != expOutDims[0] || layer.output.Height != expOutDims[1] ||
			layer.output.Depth != expOutDims[2] {
			t.Errorf("test %d gave %d,%d,%d output (expected %d,%d,%d)", i,
				layer.output.Width, layer.output.Height, layer.output.Depth,
				expOutDims[0], expOutDims[1], expOutDims[2])
		}
	}
}

func TestConvForward(t *testing.T) {
	layer := testingConvLayer()

	// Make sure that older forward propagations
	// do not affect future ones.
	input := layer.Input()
	backup := input[13]
	input[13] = 3.141592
	layer.PropagateForward()
	input[13] = backup

	layer.PropagateForward()

	expectedConvs := &Tensor3{
		Width:  2,
		Height: 3,
		Depth:  2,
		Data: []float64{
			3.175686000, 3.016690000, 2.517999000, 2.025792000,
			3.110512000, 2.324176000, 4.029134000, 3.228145000,
			3.488844000, 2.623714000, 3.571243000, 2.241387000,
		},
	}

	if expectedConvs.Width != layer.output.Width || expectedConvs.Height != layer.output.Height ||
		expectedConvs.Depth != layer.output.Depth {
		t.Fatalf("unexpected layer output dimensions: %d,%d,%d (expected %d,%d,%d)",
			layer.output.Width, layer.output.Height, layer.output.Depth,
			expectedConvs.Width, expectedConvs.Height, expectedConvs.Depth)
	}

	for y := 0; y < expectedConvs.Height; y++ {
		for x := 0; x < expectedConvs.Width; x++ {
			for z := 0; z < expectedConvs.Depth; z++ {
				expected := expectedConvs.Get(x, y, z)
				actual := layer.convolutions.Get(x, y, z)
				if math.Abs(actual-expected) > 1e-6 {
					t.Errorf("expected %f at %d,%d,%d but got %f", expected, x, y, z, actual)
				}
				expectedOut := 1 / (1 + math.Exp(-expected))
				actualOut := layer.output.Get(x, y, z)
				if math.Abs(actualOut-expectedOut) > 1e-6 {
					t.Errorf("expected output %f at %d,%d,%d but got %f", expectedOut,
						x, y, z, actualOut)
				}
			}
		}
	}
}

func testingConvLayer() *ConvLayer {
	layer := NewConvLayer(&ConvParams{
		FilterCount:  2,
		FilterWidth:  2,
		FilterHeight: 3,
		Stride:       2,
		InputWidth:   5,
		InputHeight:  7,
		InputDepth:   2,
		Activation:   Sigmoid{},
	})

	layer.SetInput([]float64{
		0.820, 0.548, 0.005, 0.850, 0.589, 0.882, 0.185, 0.243, 0.432, 0.734,
		0.478, 0.442, 0.835, 0.400, 0.270, 0.816, 0.467, 0.012, 0.060, 0.241,
		0.821, 0.069, 0.448, 0.691, 0.735, 0.090, 0.824, 0.042, 0.657, 0.707,
		0.218, 0.804, 0.025, 0.650, 0.833, 0.763, 0.788, 0.953, 0.796, 0.500,
		0.620, 0.038, 0.702, 0.524, 0.512, 0.699, 0.831, 0.122, 0.945, 0.840,
		0.584, 0.566, 0.586, 0.560, 0.109, 0.577, 0.785, 0.908, 0.080, 0.763,
		0.430, 0.561, 0.474, 0.516, 0.508, 0.062, 0.126, 0.371, 0.422, 0.424,
	})

	copy(layer.filters[0].Data, []float64{
		0.348, 0.299, 0.946, 0.806,
		0.101, 0.705, 0.821, 0.819,
		0.106, 0.348, 0.285, 0.133,
	})
	copy(layer.filters[1].Data, []float64{
		0.293, 0.494, 0.148, 0.758,
		0.901, 0.050, 0.415, 0.892,
		0.736, 0.458, 0.465, 0.167,
	})
	copy(layer.biases, []float64{0.333, -0.255})

	return layer
}
