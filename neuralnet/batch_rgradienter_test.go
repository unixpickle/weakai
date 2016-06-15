package neuralnet

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const (
	batchRGradienterSeed     = 123123
	batchRGradienterTestPrec = 1e-5
)

func TestBatchRGradienterSimple(t *testing.T) {
	testBatchRGradienter(t, 1, &BatchRGradienter{
		CostFunc:      MeanSquaredCost{},
		MaxGoroutines: 1,
		MaxBatchSize:  1,
	})
}

func TestBatchRGradienterUnevenBatch(t *testing.T) {
	testBatchRGradienter(t, 16, &BatchRGradienter{
		CostFunc:      MeanSquaredCost{},
		MaxGoroutines: 1,
		MaxBatchSize:  3,
	})
}

func TestBatchRGradienterUnevenConcurrent(t *testing.T) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(8)
	testBatchRGradienter(t, 16, &BatchRGradienter{
		CostFunc:      MeanSquaredCost{},
		MaxGoroutines: 8,
		MaxBatchSize:  1,
	})
	runtime.GOMAXPROCS(n)
}

func testBatchRGradienter(t *testing.T, batchSize int, b *BatchRGradienter) {
	rand.Seed(batchRGradienterSeed)

	net := Network{
		&DenseLayer{
			InputCount:  10,
			OutputCount: 30,
		},
		&Sigmoid{},
		&DenseLayer{
			InputCount:  30,
			OutputCount: 3,
		},
		&Sigmoid{},
	}
	net.Randomize()
	b.Learner = net.BatchLearner()

	inputs := make([]linalg.Vector, batchSize)
	outputs := make([]linalg.Vector, batchSize)
	for i := range inputs {
		inputVec := make(linalg.Vector, 10)
		outputVec := make(linalg.Vector, 3)
		for j := range inputVec {
			inputVec[j] = rand.NormFloat64()
		}
		for j := range outputVec {
			outputVec[j] = rand.Float64()
		}
		inputs[i] = inputVec
		outputs[i] = outputVec
	}
	samples := VectorSampleSet(inputs, outputs)

	rVector := autofunc.RVector(autofunc.NewGradient(net.Parameters()))
	for _, vec := range rVector {
		for i := range vec {
			vec[i] = rand.NormFloat64()
		}
	}

	single := SingleRGradienter{Learner: net, CostFunc: b.CostFunc}
	expectedGrad := single.Gradient(samples)
	actualGrad := b.Gradient(samples)

	if !vecMapsEqual(expectedGrad, actualGrad) {
		t.Error("bad gradient from Gradient()")
	}

	expectedGrad, expectedRGrad := single.RGradient(rVector, samples)
	actualGrad, actualRGrad := b.RGradient(rVector, samples)

	if !vecMapsEqual(expectedGrad, actualGrad) {
		t.Error("bad gradient from RGradient()")
	}
	if !vecMapsEqual(expectedRGrad, actualRGrad) {
		t.Error("bad r-gradient from RGradient()")
	}
}

func vecMapsEqual(m1, m2 map[*autofunc.Variable]linalg.Vector) bool {
	for k := range m1 {
		if _, ok := m2[k]; !ok {
			return false
		}
	}
	for k := range m2 {
		if _, ok := m1[k]; !ok {
			return false
		}
	}

	for k, v := range m1 {
		v1 := m2[k]
		if len(v) != len(v1) {
			return false
		}
		for i, x := range v {
			if math.Abs(x-v1[i]) > batchRGradienterTestPrec {
				return false
			}
		}
	}

	return true
}
