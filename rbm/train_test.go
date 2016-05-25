package rbm

import "testing"

const benchmarkLayerSize = 50
const benchmarkSampleCount = 10

func BenchmarkTrain(b *testing.B) {
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
		BatchSize:  1,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := NewRBM(50, 50)
		trainer.Train(r, samples)
	}
}
