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
	testTrainingXOR(t, 1, 1, 1, false)
}

func TestTrainingXORParallel(t *testing.T) {
	testTrainingXOR(t, 1, 3, 3, false)
}

func TestTrainingXORBatched(t *testing.T) {
	testTrainingXOR(t, 3, 1, 3, false)
}

func TestTrainingUneven(t *testing.T) {
	testTrainingXOR(t, 2, 2, 3, false)
}

func TestTrainingSingle(t *testing.T) {
	testTrainingXOR(t, 0, 0, 1, true)
}

func testTrainingXOR(t *testing.T, maxBatch, maxGos, batchSize int, single bool) {
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

	var gradienter Gradienter
	if single {
		gradienter = &SingleRGradienter{
			Learner:  net,
			CostFunc: MeanSquaredCost{},
		}
	} else {
		gradienter = &BatchRGradienter{
			Learner:       net,
			CostFunc:      MeanSquaredCost{},
			MaxGoroutines: maxGos,
			MaxBatchSize:  maxBatch,
		}
	}
	SGD(gradienter, samples, 0.9, 1000, batchSize)

	for i, sample := range samples.Inputs {
		output := net.Apply(&autofunc.Variable{sample}).Output()
		expected := samples.Outputs[i][0]
		actual := output[0]
		if math.Abs(expected-actual) > 0.08 {
			t.Errorf("expected %f for input %v but got %f", expected, sample, actual)
		}
	}
}

func BenchmarkTrainingBigSerial50(b *testing.B) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(1)
	benchmarkTrainingBig(b, 50, 100)
	runtime.GOMAXPROCS(n)
}

func BenchmarkTrainingBigParallel50(b *testing.B) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(2)
	benchmarkTrainingBig(b, 50, 100)
	runtime.GOMAXPROCS(n)
}

func BenchmarkTrainingBigSerial500(b *testing.B) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(1)
	benchmarkTrainingBig(b, 500, 100)
	runtime.GOMAXPROCS(n)
}

func BenchmarkTrainingBigParallel500(b *testing.B) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(2)
	benchmarkTrainingBig(b, 500, 100)
	runtime.GOMAXPROCS(n)
}

func BenchmarkTrainingBigSerial1000(b *testing.B) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(1)
	benchmarkTrainingBig(b, 1000, 100)
	runtime.GOMAXPROCS(n)
}

func BenchmarkTrainingBigParallel1000(b *testing.B) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(2)
	benchmarkTrainingBig(b, 1000, 100)
	runtime.GOMAXPROCS(n)
}

func benchmarkTrainingBig(b *testing.B, hiddenSize, batchSize int) {
	autofunc.DefaultVectorCache.Clear()
	runtime.GC()

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
			OutputCount: hiddenSize,
		},
		&Sigmoid{},
		&DenseLayer{
			InputCount:  hiddenSize,
			OutputCount: 10,
		},
		&Sigmoid{},
	}
	network.Randomize()
	batcher := &BatchRGradienter{
		Learner:  network,
		CostFunc: MeanSquaredCost{},
	}

	b.ResetTimer()
	SGD(batcher, samples, 0.01, b.N, batchSize)
}
