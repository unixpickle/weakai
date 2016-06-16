package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
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
	result := r.Block.Batch(in)

	var newStates []linalg.Vector
	var truncSecs [][]linalg.Vector
	for i, seq := range seqs {
		if len(seq) > 0 {
			newStates = append(newStates, result.States()[i])
			truncSecs = append(truncSecs, seq[1:])
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
