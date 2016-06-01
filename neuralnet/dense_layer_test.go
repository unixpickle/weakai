package neuralnet

import (
	"math"
	"testing"

	"github.com/unixpickle/serializer"
)

func TestDenseForward(t *testing.T) {
	layer := testingDenseLayer(t)
	input := layer.Input()

	// Make sure no old values are "remembered"
	// between forward propagations.
	backup := input[0]
	input[0] = 3.141592
	layer.PropagateForward()
	input[0] = backup

	layer.PropagateForward()

	if math.Abs(layer.Output()[0]-0.2689414214) > 1e-6 {
		t.Errorf("expected %f for output 0 but got %f", 0.2689414214,
			layer.Output()[0])
	}

	if math.Abs(layer.Output()[1]-0.880797078) > 1e-6 {
		t.Errorf("expected %f for output 1 but got %f", 0.880797078,
			layer.Output()[1])
	}
}

func TestDenseBackward(t *testing.T) {
	layer := testingDenseLayer(t)
	input := layer.Input()
	downGrad := layer.DownstreamGradient()

	// Make sure no old values are "remembered"
	// between forward propagations.
	backup := input[0]
	input[0] = 3.141592
	layer.PropagateForward()
	layer.PropagateBackward(true)
	input[0] = backup

	layer.PropagateForward()
	layer.PropagateBackward(true)

	partial0 := 0.1966119333 * downGrad[0]
	partial1 := 0.1049935854 * downGrad[1]

	weightGradient := [][]float64{
		[]float64{partial0 * input[0], partial0 * input[1], partial0 * input[2]},
		[]float64{partial1 * input[0], partial1 * input[1], partial1 * input[2]},
	}

	for i, xs := range weightGradient {
		for j, x := range xs {
			if actual := layer.weightGradient[i][j]; math.Abs(actual-x) > 1e-6 {
				t.Errorf("weight gradient %d,%d should be %f but got %f", i, j, x, actual)
			}
		}
	}

	biasGradient := []float64{partial0, partial1}
	for i, x := range biasGradient {
		if actual := layer.biasGradient[i]; math.Abs(actual-x) > 1e-6 {
			t.Errorf("bias gradient %d should be %f but got %f", i, x, actual)
		}
	}

	upstreamGradient := []float64{
		layer.weights[0][0]*partial0 + layer.weights[1][0]*partial1,
		layer.weights[0][1]*partial0 + layer.weights[1][1]*partial1,
		layer.weights[0][2]*partial0 + layer.weights[1][2]*partial1,
	}
	for i, x := range upstreamGradient {
		if actual := layer.UpstreamGradient()[i]; math.Abs(actual-x) > 1e-6 {
			t.Errorf("upstream gradient %d should be %f but got %f", i, x, actual)
		}
	}
}

func TestDenseSerialize(t *testing.T) {
	layer := testingDenseLayer(t)
	encoded, err := layer.Serialize()
	if err != nil {
		t.Fatal(err)
	}
	layerType := layer.SerializerType()

	decoded, err := serializer.GetDeserializer(layerType)(encoded)
	if err != nil {
		t.Fatal(err)
	}

	layer, ok := decoded.(*DenseLayer)
	if !ok {
		t.Fatalf("decoded layer was not a *DenseLayer, but a %T", decoded)
	}

	layer.SetDownstreamGradient([]float64{0.5, -0.3})
	layer.SetInput([]float64{1, -1, 2})

	layer.PropagateForward()
	layer.PropagateBackward(true)

	copy(layer.weights[0], []float64{1, 2, 3})
	copy(layer.weights[1], []float64{-3, 2, -1})
	copy(layer.biases, []float64{-6, 9})

	expLists := [][]float64{
		{1, 2, 3},
		{-3, 2, -1},
		{-6, 9},
	}
	actualLists := [][]float64{layer.weights[0], layer.weights[1], layer.biases}

	for i, x := range expLists {
		actual := actualLists[i]
		equal := true
		for j, v := range x {
			if actual[j] != v {
				equal = false
			}
		}
		if !equal {
			t.Errorf("list %d does not match", i)
		}
	}
}

func testingDenseLayer(t *testing.T) *DenseLayer {
	params := &DenseParams{
		Activation:  Sigmoid{},
		InputCount:  3,
		OutputCount: 2,
	}

	layer := NewDenseLayer(params)

	layer.SetDownstreamGradient([]float64{0.5, -0.3})
	layer.SetInput([]float64{1, -1, 2})

	if len(layer.weights) != 2 {
		t.Fatal("expected 2 weights slices but got", len(layer.weights))
	}
	if len(layer.weights[0]) != 3 {
		t.Fatal("expected 3 weights per neuron but got", len(layer.weights[0]))
	}

	copy(layer.weights[0], []float64{1, 2, 3})
	copy(layer.weights[1], []float64{-3, 2, -1})
	copy(layer.biases, []float64{-6, 9})

	return layer
}
