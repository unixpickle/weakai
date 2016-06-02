package lstm

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

func BenchmarkTrainSynchronously(b *testing.B) {
	inSeqs := make([][]linalg.Vector, 10)
	outSeqs := make([][]linalg.Vector, len(inSeqs))
	for i := range inSeqs {
		inSeqs[i] = make([]linalg.Vector, 20)
		outSeqs[i] = make([]linalg.Vector, len(inSeqs[i]))
		for j := range inSeqs[i] {
			inSeqs[i][j] = make(linalg.Vector, 15)
			outSeqs[i][j] = make(linalg.Vector, len(inSeqs[i][j]))
			idx := rand.Intn(len(inSeqs[i][j]))
			inSeqs[i][j][idx] = 1
			outSeqs[i][j][idx] = 1
		}
	}
	trainer := rnn.SGD{
		InSeqs:   inSeqs,
		OutSeqs:  outSeqs,
		CostFunc: rnn.MeanSquaredCost{},
		Epochs:   b.N * 2,
		StepSize: 0.01,
	}
	net := NewNet(rnn.Sigmoid{}, len(inSeqs[0][0]), 20, len(outSeqs[0][0]))
	b.ResetTimer()
	trainer.Train(net)
}
