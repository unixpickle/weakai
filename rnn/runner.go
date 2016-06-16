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
