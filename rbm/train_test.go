package rbm

import (
	"runtime"
	"testing"
)

const benchmarkLayerSize = 50
const benchmarkSampleCount = 10

func BenchmarkTrain(b *testing.B) {
	benchmarkTrain(b, 1)
}

func BenchmarkTrainConcurrent(b *testing.B) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(10)
	benchmarkTrain(b, 10)
	runtime.GOMAXPROCS(n)
}

func benchmarkTrain(b *testing.B, batchSize int) {
	samples := make([][]bool, benchmarkSampleCount)
	for i := range samples {
		s := make([]bool, benchmarkLayerSize)
		for j := range s {
			// Some kind of psuedo-random nonsense,
			// just to enforce determinism.
			if (i*j*3+17)%19 < 7 {
				s[j] = true
			}
		}
		samples[i] = s
	}
	trainer := Trainer{
		GibbsSteps: 10,
		StepSize:   0.01,
		Epochs:     5,
		BatchSize:  batchSize,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := NewRBM(50, 50)
		trainer.Train(r, samples)
	}
}
