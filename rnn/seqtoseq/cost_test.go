package seqtoseq

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

func TestTotalCostBlock(t *testing.T) {
	var samples sgd.SliceSampleSet
	for i := 0; i < 100; i++ {
		size := rand.Intn(10)
		inSeq := make([]linalg.Vector, size)
		outSeq := make([]linalg.Vector, size)
		for i := range inSeq {
			inSeq[i] = linalg.RandVector(3)
			outSeq[i] = linalg.RandVector(4)
		}
		samples = append(samples, Sample{Inputs: inSeq, Outputs: outSeq})
	}
	block := rnn.NewLSTM(3, 4)
	actual := TotalCostBlock(block, 7, samples, neuralnet.MeanSquaredCost{})
	expected := TotalCostSeqFunc(&rnn.BlockSeqFunc{B: block},
		7, samples, neuralnet.MeanSquaredCost{})
	if math.Abs(actual-expected) > 1e-5 {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}
