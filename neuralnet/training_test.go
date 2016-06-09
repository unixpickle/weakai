package neuralnet

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

func TestTrainingXORSerial(t *testing.T) {
	testTrainingXOR(t, 1)
}

func TestTrainingXORParallel(t *testing.T) {
	testTrainingXOR(t, 3)
}

func testTrainingXOR(t *testing.T, batchSize int) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	net := Network{
		&DenseLayer{
			InputCount:  2,
			OutputCount: 4,
		},
		&Sigmoid{},
		&DenseLayer{
			InputCount:  4,
			OutputCount: 1,
		},
		&Sigmoid{},
	}
	rand.Seed(123123)
	net.Randomize()

	samples := &SampleSet{
		Inputs: []linalg.Vector{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1},
		},
		Outputs: []linalg.Vector{{0}, {1}, {1}, {0}},
	}

	batcher := NewBatcher(net, MeanSquaredCost{}, batchSize)
	batcher.Start()
	SGD(batcher, samples, 0.9, 100000)
	batcher.Stop()

	for i, sample := range samples.Inputs {
		output := net.Apply(&autofunc.Variable{sample}).Output()
		expected := samples.Outputs[i][0]
		actual := output[0]
		if math.Abs(expected-actual) > 2e-2 {
			t.Errorf("expected %f for input %v but got %f", expected, sample, actual)
		}
	}
}

func BenchmarkTrainingBigSerial(b *testing.B) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(1)
	benchmarkTrainingBig(b, 3)
	runtime.GOMAXPROCS(n)
}

func BenchmarkTrainingBigParallel(b *testing.B) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(3)
	benchmarkTrainingBig(b, 3)
	runtime.GOMAXPROCS(n)
}

func benchmarkTrainingBig(b *testing.B, batchSize int) {
	inputs := make([]linalg.Vector, 100)
	outputs := make([]linalg.Vector, len(inputs))
	for i := range inputs {
		inputs[i] = make(linalg.Vector, 1000)
		outputs[i] = make(linalg.Vector, len(inputs[i]))
		for j := range inputs[i] {
			inputs[i][j] = rand.Float64()
			outputs[i][j] = rand.Float64()
		}
	}

	samples := &SampleSet{Inputs: inputs, Outputs: outputs}
	network := Network{
		&DenseLayer{
			InputCount:  len(inputs[0]),
			OutputCount: 500,
		},
		&Sigmoid{},
		&DenseLayer{
			InputCount:  500,
			OutputCount: 10,
		},
		&Sigmoid{},
	}
	network.Randomize()
	batcher := NewBatcher(network, MeanSquaredCost{}, batchSize)
	batcher.Start()
	defer batcher.Stop()

	b.ResetTimer()
	SGD(batcher, samples, 0.01, b.N)
}
