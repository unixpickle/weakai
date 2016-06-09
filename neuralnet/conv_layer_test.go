package neuralnet

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

func TestConvDimensions(t *testing.T) {
	layers := []*ConvLayer{
		{1, 3, 3, 1, 9, 9, 1, nil, nil, nil, nil},
		{5, 4, 7, 2, 17, 56, 18, nil, nil, nil, nil},
	}

	outputDims := [][]int{
		{7, 7},
		{7, 25},
	}

	for i, layer := range layers {
		expOutDims := outputDims[i]
		if layer.OutputWidth() != expOutDims[0] || layer.OutputHeight() != expOutDims[1] {
			t.Errorf("test %d gave %d,%d output (expected %d,%d)", i,
				layer.OutputWidth(), layer.OutputHeight(),
				expOutDims[0], expOutDims[1])
		}
	}
}

func TestConvForward(t *testing.T) {
	network, input, _ := convLayerTestInfo()

	actualConvs := network[0].Apply(input).Output()
	expectedConvs := []float64{
		3.175686000, 3.016690000, 2.517999000, 2.025792000,
		3.110512000, 2.324176000, 4.029134000, 3.228145000,
		3.488844000, 2.623714000, 3.571243000, 2.241387000,
	}

	if len(expectedConvs) != len(actualConvs) {
		t.Fatalf("unexpected layer output size: %d (expected %d)",
			len(actualConvs), len(expectedConvs))
	}
}

func TestConvBackward(t *testing.T) {
	layer, input, outGrad := convLayerTestInfo()

	actualGrad := autofunc.NewGradient(append(layer.Parameters(), input))
	layer.Apply(input).PropagateGradient(outGrad, actualGrad)

	convLayer := layer[0].(*ConvLayer)
	expectedGrad := autofunc.Gradient{
		convLayer.FilterVars[0]: []float64{
			9.181420449e-02, 6.070772494e-02, 4.831743717e-02, 6.140456075e-02,
			4.619374891e-02, 9.677697371e-02, 5.711791144e-02, 5.124701355e-02,
			8.690832544e-02, 2.255616739e-02, 9.041001878e-02, 4.383411433e-02,
		},
		convLayer.FilterVars[1]: []float64{
			1.725619176e-01, 1.501485079e-01, 1.396596513e-01, 8.822688174e-02,
			1.043560711e-01, 1.851411351e-01, 1.769153948e-01, 1.366024735e-01,
			1.678136736e-01, 6.694391158e-02, 1.517132408e-01, 8.335992965e-02,
		},
		convLayer.Biases: []float64{1.333355836e-01, 2.790278869e-01},
		input: []float64{
			1.346240470e-02, 1.840140585e-02, 1.830078429e-02, 3.341979500e-02, 4.527417587e-02, 6.139417717e-02, 6.285708549e-02, 1.122305051e-01, 0, 0,
			2.692730031e-02, 1.193745091e-02, 2.396698285e-02, 3.739434288e-02, 8.890665566e-02, 4.124498873e-02, 8.115978953e-02, 1.253480957e-01, 0, 0,
			3.633179008e-02, 3.105761526e-02, 5.291576339e-02, 3.939048624e-02, 8.488640888e-02, 7.725933595e-02, 6.877644332e-02, 5.033669814e-02, 0, 0,
			7.172645109e-03, 2.625700212e-02, 3.193879788e-02, 3.368514841e-02, 2.737903811e-02, 6.263677753e-03, 1.786440555e-02, 3.198290875e-02, 0, 0,
			2.969143512e-02, 4.797023692e-02, 3.826207676e-02, 6.320548619e-02, 4.395410081e-02, 5.088142526e-02, 2.968988521e-02, 6.090264241e-02, 0, 0,
			5.255802153e-02, 1.594788029e-02, 3.863840312e-02, 6.542970202e-02, 6.192735934e-02, 6.301981015e-03, 3.169670830e-02, 6.425452037e-02, 0, 0,
			4.337086165e-02, 3.224390653e-02, 3.146379199e-02, 1.187088457e-02, 5.068287349e-02, 3.269456802e-02, 3.291436767e-02, 1.194641079e-02, 0, 0,
		},
	}
	varNames := map[*autofunc.Variable]string{
		convLayer.FilterVars[0]: "filter 0",
		convLayer.FilterVars[1]: "filter 1",
		convLayer.Biases:        "biases",
		input:                   "input",
	}

	for variable, expected := range expectedGrad {
		name := varNames[variable]
		actual := actualGrad[variable]
		if len(expected) != len(actual) {
			t.Errorf("variable %s: expected len %d got len %d",
				name, len(expected), len(actual))
			continue
		}
		for i, x := range expected {
			a := actual[i]
			if math.Abs(x-a) > 1e-6 {
				t.Errorf("variable %s: value %d: expected %f got %f", name, i, x, a)
			}
		}
	}
}

func TestConvLayerSerialize(t *testing.T) {
	layer, _, _ := convLayerTestInfo()
	data, err := layer.Serialize()
	if err != nil {
		t.Fatal(err)
	}
	dataType := layer.SerializerType()

	l, err := serializer.GetDeserializer(dataType)(data)
	if err != nil {
		t.Fatal(err)
	}
	layer, ok := l.(Network)
	if !ok {
		t.Fatalf("decoded layer was not a Network, but rather a %T", l)
	}

	expLists := [][]float64{
		{
			0.348, 0.299, 0.946, 0.806,
			0.101, 0.705, 0.821, 0.819,
			0.106, 0.348, 0.285, 0.133,
		},
		{
			0.293, 0.494, 0.148, 0.758,
			0.901, 0.050, 0.415, 0.892,
			0.736, 0.458, 0.465, 0.167,
		},
		{0.333, -0.255},
	}
	actualLists := [][]float64{
		layer[0].(*ConvLayer).Filters[0].Data,
		layer[0].(*ConvLayer).Filters[1].Data,
		layer[0].(*ConvLayer).Biases.Vector,
	}

	for i, x := range expLists {
		actual := actualLists[i]
		equal := true
		for j, v := range x {
			if math.Abs(actual[j]-v) > 1e-6 {
				equal = false
			}
		}
		if !equal {
			t.Errorf("list %d does not match", i)
		}
	}
}

func TestConvLayerRProp(t *testing.T) {
	layer, input, _ := convLayerTestInfo()
	variables := append(layer.Parameters(), input)
	rVector := autofunc.RVector{}
	for _, variable := range variables {
		rVector[variable] = make(linalg.Vector, len(variable.Vector))
		for i := range rVector[variable] {
			rVector[variable][i] = rand.Float64()*2 - 1
		}
	}
	funcTest := &functest.RFuncTest{
		F:     layer,
		Vars:  variables,
		Input: autofunc.NewRVariable(input, rVector),
		RV:    rVector,
	}
	funcTest.Run(t)
}

func convLayerTestInfo() (network Network, input *autofunc.Variable, outGrad linalg.Vector) {
	layer := &ConvLayer{
		FilterCount:  2,
		FilterWidth:  2,
		FilterHeight: 3,
		Stride:       2,
		InputWidth:   5,
		InputHeight:  7,
		InputDepth:   2,
	}
	network = Network{layer, &Sigmoid{}}
	network.Randomize()

	input = &autofunc.Variable{
		Vector: linalg.Vector([]float64{
			0.820, 0.548, 0.005, 0.850, 0.589, 0.882, 0.185, 0.243, 0.432, 0.734,
			0.478, 0.442, 0.835, 0.400, 0.270, 0.816, 0.467, 0.012, 0.060, 0.241,
			0.821, 0.069, 0.448, 0.691, 0.735, 0.090, 0.824, 0.042, 0.657, 0.707,
			0.218, 0.804, 0.025, 0.650, 0.833, 0.763, 0.788, 0.953, 0.796, 0.500,
			0.620, 0.038, 0.702, 0.524, 0.512, 0.699, 0.831, 0.122, 0.945, 0.840,
			0.584, 0.566, 0.586, 0.560, 0.109, 0.577, 0.785, 0.908, 0.080, 0.763,
			0.430, 0.561, 0.474, 0.516, 0.508, 0.062, 0.126, 0.371, 0.422, 0.424,
		}),
	}

	outGrad = linalg.Vector([]float64{
		0.388, 0.634, 0.752, 0.902,
		0.905, 0.047, 0.395, 0.808,
		0.648, 0.892, 0.154, 0.786,
	})

	layer.Randomize()

	copy(layer.Filters[0].Data, []float64{
		0.348, 0.299, 0.946, 0.806,
		0.101, 0.705, 0.821, 0.819,
		0.106, 0.348, 0.285, 0.133,
	})
	copy(layer.Filters[1].Data, []float64{
		0.293, 0.494, 0.148, 0.758,
		0.901, 0.050, 0.415, 0.892,
		0.736, 0.458, 0.465, 0.167,
	})
	copy(layer.Biases.Vector, []float64{0.333, -0.255})

	return
}
