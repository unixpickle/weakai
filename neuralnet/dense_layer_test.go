package neuralnet

import (
	"encoding/json"
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

func BenchmarkDenseLayerBackProp(b *testing.B) {
	net := Network{
		&DenseLayer{InputCount: 1000, OutputCount: 2000},
		&Sigmoid{},
		&DenseLayer{InputCount: 2000, OutputCount: 512},
		&Sigmoid{},
		&DenseLayer{InputCount: 512, OutputCount: 10},
		&Sigmoid{},
	}
	rand.Seed(123)
	net.Randomize()
	inVec := &autofunc.Variable{Vector: make(linalg.Vector, 1000)}
	for i := range inVec.Vector {
		inVec.Vector[i] = rand.Float64()*2 - 1
	}

	downstream := make(linalg.Vector, 10)
	for i := range downstream {
		downstream[i] = 1
	}

	grad := autofunc.NewGradient(net.Parameters())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		net.Apply(inVec).PropagateGradient(downstream, grad)
	}
}

func BenchmarkDenseLayerSerialization(b *testing.B) {
	dl := &DenseLayer{
		InputCount:  128 * 8 * 8,
		OutputCount: 128,
	}
	dl.Randomize()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, _ := dl.Serialize()
		DeserializeDenseLayer(data)
	}
}

func TestDenseForward(t *testing.T) {
	network, input, _ := denseTestInfo()
	output := network.Apply(input)

	if math.Abs(output.Output()[0]-0.2689414214) > 1e-6 {
		t.Errorf("expected %f for output 0 but got %f", 0.2689414214,
			output.Output()[0])
	}

	if math.Abs(output.Output()[1]-0.880797078) > 1e-6 {
		t.Errorf("expected %f for output 1 but got %f", 0.880797078,
			output.Output()[1])
	}
}

func TestDenseBackward(t *testing.T) {
	network, inputVar, upstream := denseTestInfo()

	partial0 := 0.1966119333 * upstream[0]
	partial1 := 0.1049935854 * upstream[1]

	input := inputVar.Vector
	weightGradient := [][]float64{
		[]float64{partial0 * input[0], partial0 * input[1], partial0 * input[2]},
		[]float64{partial1 * input[0], partial1 * input[1], partial1 * input[2]},
	}

	weightVec := network[0].(*DenseLayer).Weights.Data.Vector
	upstreamGradient := []float64{
		weightVec[0]*partial0 + weightVec[3]*partial1,
		weightVec[1]*partial0 + weightVec[4]*partial1,
		weightVec[2]*partial0 + weightVec[5]*partial1,
	}

	params := network.Parameters()
	params = append(params, inputVar)
	actualGrad := autofunc.NewGradient(params)
	output := network.Apply(inputVar)
	output.PropagateGradient(upstream, actualGrad)

	for i, xs := range weightGradient {
		for j, x := range xs {
			actualVec := actualGrad[network[0].(*DenseLayer).Weights.Data]
			actual := actualVec[i*3+j]
			if math.Abs(actual-x) > 1e-6 {
				t.Errorf("weight gradient %d,%d should be %f but got %f", i, j, x, actual)
			}
		}
	}

	biasGradient := []float64{partial0, partial1}
	for i, x := range biasGradient {
		actualVec := actualGrad[network[0].(*DenseLayer).Biases.Var]
		if actual := actualVec[i]; math.Abs(actual-x) > 1e-6 {
			t.Errorf("bias gradient %d should be %f but got %f", i, x, actual)
		}
	}

	for i, x := range upstreamGradient {
		actualVec := actualGrad[inputVar]
		if actual := actualVec[i]; math.Abs(actual-x) > 1e-6 {
			t.Errorf("upstream gradient %d should be %f but got %f", i, x, actual)
		}
	}
}

func TestDenseSerialize(t *testing.T) {
	network, _, _ := denseTestInfo()
	layer := network[0].(*DenseLayer)

	normalEncoded, err := layer.Serialize()
	if err != nil {
		t.Fatal(err)
	}
	jsonEncoded, _ := json.Marshal(layer)
	layerType := layer.SerializerType()

	for i, encoded := range [][]byte{normalEncoded, jsonEncoded} {
		decoded, err := serializer.GetDeserializer(layerType)(encoded)
		if err != nil {
			t.Fatal(err)
		}

		layer, ok := decoded.(*DenseLayer)
		if !ok {
			t.Fatalf("%d: decoded layer was not a *DenseLayer, but a %T", i, decoded)
		}

		expLists := [][]float64{
			{1, 2, 3, -3, 2, -1},
			{-6, 9},
		}
		actualLists := [][]float64{layer.Weights.Data.Vector, layer.Biases.Var.Vector}

		for k, x := range expLists {
			actual := actualLists[k]
			equal := true
			for j, v := range x {
				if actual[j] != v {
					equal = false
				}
			}
			if !equal {
				t.Errorf("%d: list %d does not match", i, k)
			}
		}
	}

}

func denseTestInfo() (network Network, input *autofunc.Variable, grad linalg.Vector) {
	denseLayer := &DenseLayer{
		InputCount:  3,
		OutputCount: 2,
	}
	denseLayer.Randomize()
	network = Network{denseLayer, &Sigmoid{}}

	grad = linalg.Vector([]float64{0.5, -0.3})
	input = &autofunc.Variable{Vector: linalg.Vector([]float64{1, -1, 2})}

	copy(denseLayer.Weights.Data.Vector, []float64{1, 2, 3, -3, 2, -1})
	copy(denseLayer.Biases.Var.Vector, []float64{-6, 9})

	return
}
