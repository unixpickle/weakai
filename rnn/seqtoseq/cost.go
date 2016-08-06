package seqtoseq

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

// TotalCostBlock runs an rnn.Block on a set of Samples
// and evaluates the total output cost.
//
// The batchSize specifies how many samples to run in
// batches while computing the cost.
func TotalCostBlock(b rnn.Block, batchSize int, s sgd.SampleSet, c neuralnet.CostFunc) float64 {
	runner := &rnn.Runner{Block: b}

	var cost float64
	for i := 0; i < s.Len(); i += batchSize {
		var inSeqs, outSeqs [][]linalg.Vector
		for j := i; j < i+batchSize && j < s.Len(); j++ {
			seq := s.GetSample(j).(Sample)
			inSeqs = append(inSeqs, seq.Inputs)
			outSeqs = append(outSeqs, seq.Outputs)
		}
		output := runner.RunAll(inSeqs)
		for j, outSeq := range outSeqs {
			for t, actual := range output[j] {
				expected := outSeq[t]
				actualVar := &autofunc.Variable{Vector: actual}
				cost += c.Cost(expected, actualVar).Output()[0]
			}
		}
	}
	return cost
}

// TotalCostSeqFunc runs an rnn.SeqFunc on a set of
// Samples and evaluates the total output cost.
//
// The batchSize specifies how many samples to run in
// batches while computing the cost.
func TotalCostSeqFunc(f rnn.SeqFunc, batchSize int, s sgd.SampleSet,
	c neuralnet.CostFunc) float64 {
	var totalCost float64
	for i := 0; i < s.Len(); i += batchSize {
		var inSeqs [][]autofunc.Result
		var outSeqs [][]linalg.Vector
		for j := i; j < i+batchSize && j < s.Len(); j++ {
			seq := s.GetSample(j).(Sample)
			inSeq := make([]autofunc.Result, len(seq.Inputs))
			for k, in := range seq.Inputs {
				inSeq[k] = &autofunc.Variable{Vector: in}
			}
			inSeqs = append(inSeqs, inSeq)
			outSeqs = append(outSeqs, seq.Outputs)
		}
		output := f.BatchSeqs(inSeqs)
		for j, actualSeq := range output.OutputSeqs() {
			expectedSeq := outSeqs[j]
			for k, actual := range actualSeq {
				expected := expectedSeq[k]
				actualVar := &autofunc.Variable{Vector: actual}
				totalCost += c.Cost(expected, actualVar).Output()[0]
			}
		}
	}
	return totalCost
}
