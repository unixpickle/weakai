package neuralnet

import (
	"math"
	"math/rand"
	"testing"
)

const derivativeEpsilon = 1e-5

func TestGradients(t *testing.T) {
	net, err := NewNetwork([]LayerPrototype{
		&DenseParams{
			Activation:  Sigmoid{},
			InputCount:  2,
			OutputCount: 4,
		},
		&DenseParams{
			Activation:  Sigmoid{},
			InputCount:  4,
			OutputCount: 2,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	rand.Seed(123123)
	net.Randomize()

	input := []float64{1, 0}
	expectedOutput := []float64{0, 1}

	layer0 := net.Layers[0].(*DenseLayer)
	layer1 := net.Layers[1].(*DenseLayer)

	var paramPtrs [][]float64
	var paramGradPtrs [][]float64

	for _, layer := range []*DenseLayer{layer1, layer0} {
		for i, weights := range layer.Weights() {
			for j := range weights {
				paramPtrs = append(paramPtrs, weights[j:])
				paramGradPtrs = append(paramGradPtrs, layer.WeightGradients()[i][j:])
			}
		}
		for i := range layer.Biases() {
			paramPtrs = append(paramPtrs, layer.Biases()[i:])
			paramGradPtrs = append(paramGradPtrs, layer.BiasGradients()[i:])
		}
	}

	for i, paramPtr := range paramPtrs {
		gradPtr := paramGradPtrs[i]

		net.SetInput(input)
		net.PropagateForward()
		net.SetDownstreamGradient([]float64{
			net.Output()[0] - expectedOutput[0],
			net.Output()[1] - expectedOutput[1],
		})
		net.PropagateBackward(false)

		grad := gradPtr[0]
		expected := approximatePartial(net, paramPtr, expectedOutput)
		if math.Abs(expected-grad) > 1e-4 {
			t.Errorf("expected gradient %d to be %f but got %f", i, expected, grad)
		}
	}
}

func approximatePartial(n *Network, paramPtr []float64, expected []float64) float64 {
	center := paramPtr[0]
	paramPtr[0] = center - derivativeEpsilon
	n.PropagateForward()

	var out1 float64
	for i, x := range expected {
		out1 += 0.5 * math.Pow(n.Output()[i]-x, 2)
	}

	paramPtr[0] = center + derivativeEpsilon
	n.PropagateForward()
	var out2 float64
	for i, x := range expected {
		out2 += 0.5 * math.Pow(n.Output()[i]-x, 2)
	}

	paramPtr[0] = center
	return (out2 - out1) / (derivativeEpsilon * 2)
}
