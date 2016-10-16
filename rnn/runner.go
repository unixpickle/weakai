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
	currentState State
}

// Reset resets the current state, effectively going
// back to time 0.
func (r *Runner) Reset() {
	r.currentState = nil
}

// StepTime evaluates the block with the current state
// and input, updating the internal state and returning
// the output.
func (r *Runner) StepTime(input linalg.Vector) linalg.Vector {
	if r.currentState == nil {
		r.currentState = r.Block.StartState()
	}
	inVar := &autofunc.Variable{Vector: input}
	out := r.Block.ApplyBlock([]State{r.currentState}, []autofunc.Result{inVar})
	r.currentState = out.States()[0]
	return out.Outputs()[0]
}

// RunAll evaluates a batch of sequences on the block and
// returns the new output sequences.
// All of the sequences are evaluated in a single batch.
// This does not affect or use the state used by Reset and
// StepTime.
func (r *Runner) RunAll(inputs [][]linalg.Vector) [][]linalg.Vector {
	initState := r.Block.StartState()
	initStates := make([]State, len(inputs))
	for i := range initStates {
		initStates[i] = initState
	}
	return r.recursiveRunAll(inputs, initStates)
}

func (r *Runner) recursiveRunAll(seqs [][]linalg.Vector, states []State) [][]linalg.Vector {
	var inRes []autofunc.Result
	var inStates []State
	for i, seq := range seqs {
		if len(seq) > 0 {
			inVar := &autofunc.Variable{Vector: seq[0]}
			inRes = append(inRes, inVar)
			inStates = append(inStates, states[i])
		}
	}
	if len(inRes) == 0 {
		return make([][]linalg.Vector, len(seqs))
	}
	result := r.Block.ApplyBlock(inStates, inRes)

	var newStates []State
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
