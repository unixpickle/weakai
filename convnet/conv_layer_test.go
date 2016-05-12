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

func TestConvBackward(t *testing.T) {
	layer := testingConvLayer()

	// Make sure that older forward propagations
	// do not affect future ones.
	input := layer.Input()
	backup := input[13]
	input[13] = 3.141592
	layer.PropagateForward()
	layer.PropagateBackward()
	input[13] = backup

	layer.PropagateForward()
	layer.PropagateBackward()

	filterGradients := []*Tensor3{
		{
			Width:  2,
			Height: 3,
			Depth:  2,
			Data: []float64{
				9.181420449e-02, 6.070772494e-02, 4.831743717e-02, 6.140456075e-02,
				4.619374891e-02, 9.677697371e-02, 5.711791144e-02, 5.124701355e-02,
				8.690832544e-02, 2.255616739e-02, 9.041001878e-02, 4.383411433e-02,
			},
		},
		{
			Width:  2,
			Height: 3,
			Depth:  2,
			Data: []float64{
				1.725619176e-01, 1.501485079e-01, 1.396596513e-01, 8.822688174e-02,
				1.043560711e-01, 1.851411351e-01, 1.769153948e-01, 1.366024735e-01,
				1.678136736e-01, 6.694391158e-02, 1.517132408e-01, 8.335992965e-02,
			},
		},
	}

	for i, expected := range filterGradients {
		actual := layer.filterGradients[i]
		for j, exp := range expected.Data {
			if math.IsNaN(actual.Data[j]) || math.Abs(actual.Data[j]-exp) > 1e-6 {
				t.Errorf("invalid partial %d,(%d) got %f expected %f",
					i, j, actual.Data[j], exp)
			}
		}
	}

	biasGradients := []float64{1.333355836e-01, 2.790278869e-01}
	if len(biasGradients) != len(layer.biasPartials) {
		t.Fatal("invalid bias partial size")
	}
	for i, x := range biasGradients {
		actual := layer.biasPartials[i]
		if math.IsNaN(actual) || math.Abs(actual-x) > 1e-6 {
			t.Errorf("bias gradient %d should be %f but got %f", i, x, actual)
		}
	}

	inputPartial := &Tensor3{
		Width:  5,
		Height: 7,
		Depth:  2,
		Data: []float64{
			1.346240470e-02, 1.840140585e-02, 1.830078429e-02, 3.341979500e-02, 4.527417587e-02, 6.139417717e-02, 6.285708549e-02, 1.122305051e-01, 0, 0,
			2.692730031e-02, 1.193745091e-02, 2.396698285e-02, 3.739434288e-02, 8.890665566e-02, 4.124498873e-02, 8.115978953e-02, 1.253480957e-01, 0, 0,
			3.633179008e-02, 3.105761526e-02, 5.291576339e-02, 3.939048624e-02, 8.488640888e-02, 7.725933595e-02, 6.877644332e-02, 5.033669814e-02, 0, 0,
			7.172645109e-03, 2.625700212e-02, 3.193879788e-02, 3.368514841e-02, 2.737903811e-02, 6.263677753e-03, 1.786440555e-02, 3.198290875e-02, 0, 0,
			2.969143512e-02, 4.797023692e-02, 3.826207676e-02, 6.320548619e-02, 4.395410081e-02, 5.088142526e-02, 2.968988521e-02, 6.090264241e-02, 0, 0,
			5.255802153e-02, 1.594788029e-02, 3.863840312e-02, 6.542970202e-02, 6.192735934e-02, 6.301981015e-03, 3.169670830e-02, 6.425452037e-02, 0, 0,
			4.337086165e-02, 3.224390653e-02, 3.146379199e-02, 1.187088457e-02, 5.068287349e-02, 3.269456802e-02, 3.291436767e-02, 1.194641079e-02, 0, 0,
		},
	}

	for y := 0; y < inputPartial.Height; y++ {
		for x := 0; x < inputPartial.Width; x++ {
			for z := 0; z < inputPartial.Depth; z++ {
				actual := layer.upstreamGradient.Get(x, y, z)
				expected := inputPartial.Get(x, y, z)
				if math.IsNaN(actual) || math.Abs(actual-expected) > 1e-6 {
					t.Errorf("gradient %d,%d,%d should be %f but got %f",
						x, y, z, expected, actual)
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

	layer.SetDownstreamGradient([]float64{
		0.388, 0.634, 0.752, 0.902,
		0.905, 0.047, 0.395, 0.808,
		0.648, 0.892, 0.154, 0.786,
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
