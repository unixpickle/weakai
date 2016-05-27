package neuralnet

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

func TestTrainingXORSerial(t *testing.T) {
	testTrainingXOR(t, 1)
}

func TestTrainingXORParallel(t *testing.T) {
	testTrainingXOR(t, 3)
}

func testTrainingXOR(t *testing.T, batchSize int) {
	net, err := NewNetwork([]LayerPrototype{
		&DenseParams{
			Activation:  Sigmoid{},
			InputCount:  2,
			OutputCount: 4,
		},
		&DenseParams{
			Activation:  Sigmoid{},
			InputCount:  4,
			OutputCount: 1,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	trainer := &SGD{
		CostFunc: MeanSquaredCost{},
		Inputs: [][]float64{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1},
		},
		Outputs:   [][]float64{{0}, {1}, {1}, {0}},
		StepSize:  0.9,
		Epochs:    100000,
		BatchSize: batchSize,
	}

	rand.Seed(123123)
	net.Randomize()
	trainer.Train(net)

	for i, sample := range trainer.Inputs {
		net.SetInput(sample)
		net.PropagateForward()
		expected := trainer.Outputs[i][0]
		actual := net.Output()[0]
		if math.Abs(expected-actual) > 2e-2 {
			t.Errorf("expected %f for input %v but got %f", expected, sample, actual)
		}
	}
}

func BenchmarkTrainingBigSerial(b *testing.B) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(1)
	benchmarkTrainingBig(b)
	runtime.GOMAXPROCS(n)
}

func BenchmarkTrainingBigParallel(b *testing.B) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(3)
	benchmarkTrainingBig(b)
	runtime.GOMAXPROCS(n)
}

func benchmarkTrainingBig(b *testing.B) {
	inputs := make([][]float64, 100)
	outputs := make([][]float64, len(inputs))
	for i := range inputs {
		inputs[i] = make([]float64, 1000)
		outputs[i] = make([]float64, len(inputs[i]))
		for j := range inputs[i] {
			inputs[i][j] = rand.Float64()
			outputs[i][j] = rand.Float64()
		}
	}
	trainer := &SGD{
		CostFunc:  MeanSquaredCost{},
		Inputs:    inputs,
		Outputs:   outputs,
		StepSize:  0.01,
		Epochs:    10,
		BatchSize: runtime.GOMAXPROCS(0),
	}

	prototype := []LayerPrototype{
		&DenseParams{
			Activation:  Sigmoid{},
			InputCount:  len(inputs[0]),
			OutputCount: 50,
		},
		&DenseParams{
			Activation:  Sigmoid{},
			InputCount:  50,
			OutputCount: 10,
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, _ := NewNetwork(prototype)
		trainer.Train(n)
	}
}
