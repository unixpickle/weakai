package seqtoseq

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
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
// If it is 0, the whole thing is run in one batch.
func TotalCostBlock(b rnn.Block, batchSize int, s sgd.SampleSet, c neuralnet.CostFunc) float64 {
	if batchSize == 0 {
		return blockBatchCost(b, s, c)
	}
	var cost float64
	for i := 0; i < s.Len(); i += batchSize {
		bs := batchSize
		if bs > s.Len()-i {
			bs = s.Len() - i
		}
		cost += blockBatchCost(b, s.Subset(i, i+bs), c)
	}
	return cost
}

// TotalCostSeqFunc runs a seqfunc.RFunc on a set of
// Samples and evaluates the total output cost.
//
// The batchSize specifies how many samples to run in
// batches while computing the cost.
func TotalCostSeqFunc(f seqfunc.RFunc, batchSize int, s sgd.SampleSet,
	c neuralnet.CostFunc) float64 {
	var totalCost float64
	for i := 0; i < s.Len(); i += batchSize {
		var inSeqs [][]linalg.Vector
		var outSeqs [][]linalg.Vector
		for j := i; j < i+batchSize && j < s.Len(); j++ {
			seq := s.GetSample(j).(Sample)
			inSeqs = append(inSeqs, seq.Inputs)
			outSeqs = append(outSeqs, seq.Outputs)
		}
		output := f.ApplySeqs(seqfunc.ConstResult(inSeqs))
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

func blockBatchCost(b rnn.Block, s sgd.SampleSet, c neuralnet.CostFunc) float64 {
	states := make([]rnn.State, s.Len())
	inputs := make([][]linalg.Vector, s.Len())
	outputs := make([][]linalg.Vector, s.Len())
	maxLen := 0
	for i := range states {
		states[i] = b.StartState()
		s := s.GetSample(i).(Sample)
		inputs[i] = s.Inputs
		outputs[i] = s.Outputs
		if len(s.Inputs) > maxLen {
			maxLen = len(s.Inputs)
		}
	}
	var totalCost float64
	for i := 0; i < maxLen; i++ {
		inStates := make([]rnn.State, 0, s.Len())
		ins := make([]autofunc.Result, 0, s.Len())
		outs := make([]linalg.Vector, 0, s.Len())
		for j, x := range inputs {
			if len(x) > 0 {
				ins = append(ins, &autofunc.Variable{Vector: x[0]})
				outs = append(outs, outputs[j][0])
				inStates = append(inStates, states[j])
			}
		}
		result := b.ApplyBlock(inStates, ins)
		for j, out := range result.Outputs() {
			totalCost += c.Cost(outs[j], &autofunc.Variable{Vector: out}).Output()[0]
		}
		var stateIdx int
		for j, x := range inputs {
			if len(x) > 0 {
				inputs[j] = x[1:]
				outputs[j] = outputs[j][1:]
				states[j] = result.States()[stateIdx]
				stateIdx++
			}
		}
	}
	return totalCost
}
