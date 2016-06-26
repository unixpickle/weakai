package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// Runner makes it easy to evaluate an RNN across time
// without having to manually store states and build
// BlockInputs.
type Runner struct {
	Block        Block
	currentState linalg.Vector
}

// Reset resets the current state, effectively going
// back to time 0.
func (r *Runner) Reset() {
	r.currentState = nil
}

// StepTime evaluates the RNN with the current state
// and input, updating the internal state and returning
// the output.
func (r *Runner) StepTime(input linalg.Vector) linalg.Vector {
	if r.currentState == nil {
		r.currentState = make(linalg.Vector, r.Block.StateSize())
	}
	in := &BlockInput{
		States: []*autofunc.Variable{
			&autofunc.Variable{Vector: r.currentState},
		},
		Inputs: []*autofunc.Variable{
			&autofunc.Variable{Vector: input},
		},
	}
	res := r.Block.Batch(in)
	r.currentState = res.States()[0]
	return res.Outputs()[0]
}

// RunAll evaluates a series of sequences on the RNN
// at once and returns the corresponding output
// sequences.
// All of the sequences are evaluated simultaneously
// using the block's Batch method.
// This does not affect or use the state that Reset
// and StepTime operate on.
func (r *Runner) RunAll(inputs [][]linalg.Vector) [][]linalg.Vector {
	zeroState := make(linalg.Vector, r.Block.StateSize())
	initStates := make([]linalg.Vector, len(inputs))
	for i := range initStates {
		initStates[i] = zeroState
	}
	return r.recursiveRunAll(inputs, initStates)
}

// TotalCost evaluates a cost function for every input
// in a set of Sequences and returns the total cost.
//
// The batchSize specifies how many samples to run
// in batches while computing the cost.
func (r *Runner) TotalCost(batchSize int, s sgd.SampleSet, c neuralnet.CostFunc) float64 {
	var cost float64
	for i := 0; i < s.Len(); i += batchSize {
		var inSeqs, outSeqs [][]linalg.Vector
		for j := i; j < i+batchSize && j < s.Len(); j++ {
			seq := s.GetSample(j).(Sequence)
			inSeqs = append(inSeqs, seq.Inputs)
			outSeqs = append(outSeqs, seq.Outputs)
		}
		output := r.RunAll(inSeqs)
		for j, outSeq := range outSeqs {
			for t, actual := range output[j] {
				expected := outSeq[t]
				actualVar := &autofunc.Variable{actual}
				cost += c.Cost(expected, actualVar).Output()[0]
			}
		}
	}
	return cost
}

func (r *Runner) recursiveRunAll(seqs [][]linalg.Vector,
	states []linalg.Vector) [][]linalg.Vector {
	in := &BlockInput{}
	for i, seq := range seqs {
		if len(seq) > 0 {
			stateVar := &autofunc.Variable{Vector: states[i]}
			inVar := &autofunc.Variable{Vector: seq[0]}
			in.States = append(in.States, stateVar)
			in.Inputs = append(in.Inputs, inVar)
		}
	}
	if len(in.Inputs) == 0 {
		return make([][]linalg.Vector, len(seqs))
	}
	result := r.Block.Batch(in)

	var newStates []linalg.Vector
	var truncSecs [][]linalg.Vector
	usedIdx := 0
	for _, seq := range seqs {
		if len(seq) > 0 {
			newStates = append(newStates, result.States()[usedIdx])
			truncSecs = append(truncSecs, seq[1:])
			usedIdx++
		}
	}
	nextOutput := r.recursiveRunAll(truncSecs, newStates)

	var resVecs [][]linalg.Vector
	nextIdx := 0
	for _, seq := range seqs {
		if len(seq) > 0 {
			nextOut := nextOutput[nextIdx]
			thisOut := append([]linalg.Vector{result.Outputs()[nextIdx]}, nextOut...)
			resVecs = append(resVecs, thisOut)
			nextIdx++
		} else {
			resVecs = append(resVecs, nil)
		}
	}

	return resVecs
}
