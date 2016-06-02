package lstm

import (
	"math/rand"
	"runtime"
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

func BenchmarkTrainSynchronously(b *testing.B) {
	benchmarkTrain(b, true)
}

func BenchmarkTrainAsynchronously(b *testing.B) {
	benchmarkTrain(b, false)
}

func benchmarkTrain(b *testing.B, sync bool) {
	inSeqs := make([][]linalg.Vector, 10)
	outSeqs := make([][]linalg.Vector, len(inSeqs))
	for i := range inSeqs {
		inSeqs[i] = make([]linalg.Vector, 30)
		outSeqs[i] = make([]linalg.Vector, len(inSeqs[i]))
		for j := range inSeqs[i] {
			inSeqs[i][j] = make(linalg.Vector, 40)
			outSeqs[i][j] = make(linalg.Vector, len(inSeqs[i][j]))
			idx := rand.Intn(len(inSeqs[i][j]))
			inSeqs[i][j][idx] = 1
			outSeqs[i][j][idx] = 1
		}
	}
	trainer := rnn.SGD{
		InSeqs:    inSeqs,
		OutSeqs:   outSeqs,
		CostFunc:  rnn.MeanSquaredCost{},
		Epochs:    b.N,
		StepSize:  0.01,
		BatchSize: 10,
	}
	net := NewNet(rnn.Sigmoid{}, len(inSeqs[0][0]), 20, len(outSeqs[0][0]))
	if sync {
		b.ResetTimer()
		trainer.TrainSynchronously(net)
	} else {
		n := runtime.GOMAXPROCS(0)
		runtime.GOMAXPROCS(4)
		b.ResetTimer()
		trainer.Train(net)
		runtime.GOMAXPROCS(n)
	}
}
