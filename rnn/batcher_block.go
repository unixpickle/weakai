package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// BatcherBlock creates a Block from an autofunc.RBatcher.
//
// The inputs and outputs of the RBatcher are packed with
// the state appended to the end of the input/output.
// For example, an input of {1,2} and a state of {3,4}
// would be packed as {1,2,3,4}.
type BatcherBlock struct {
	B         autofunc.RBatcher
	StateSize int

	// Start is the initial state.
	// If it is nil, a zero initial state is used.
	Start *autofunc.Variable
}

// StartState returns the initial state.
// This is either b.Start, or a 0 vector.
func (b *BatcherBlock) StartState() State {
	if b.Start != nil {
		return VecState(b.Start.Vector)
	}
	return VecState(make(linalg.Vector, b.StateSize))
}

// StartStateR is like StartState but with an RState.
func (b *BatcherBlock) StartStateR(rv autofunc.RVector) RState {
	if b.Start != nil {
		rVar := autofunc.NewRVariable(b.Start, rv)
		return VecRState{
			State:  rVar.Output(),
			RState: rVar.ROutput(),
		}
	}
	zero := make(linalg.Vector, b.StateSize)
	return VecRState{
		State:  zero,
		RState: zero,
	}
}

// ApplyBlock applies the batcher to the inputs.
func (b *BatcherBlock) ApplyBlock(s []State, in []autofunc.Result) BlockResult {
	res := &batcherBlockResult{
		StateSize: b.StateSize,
		Ins:       in,
	}
	var pool linalg.Vector
	for i, state := range s {
		inVec := in[i].Output()
		pool = append(pool, inVec...)
		pool = append(pool, state.(VecState)...)
	}
	res.InPool = &autofunc.Variable{Vector: pool}
	res.BatcherOut = b.B.Batch(res.InPool, len(in))

	outs, states := splitBatcherOuts(len(in), b.StateSize, res.BatcherOut.Output())
	res.OutVecs = outs
	res.OutStates = make([]State, len(states))
	for i, x := range states {
		res.OutStates[i] = VecState(x)
	}

	return res
}

// ApplyBlockR is like ApplyBlock with RResults.
func (b *BatcherBlock) ApplyBlockR(rv autofunc.RVector, s []RState,
	in []autofunc.RResult) BlockRResult {
	// TODO: this.
	return nil
}

type batcherBlockResult struct {
	StateSize  int
	Ins        []autofunc.Result
	InPool     *autofunc.Variable
	BatcherOut autofunc.Result
	OutVecs    []linalg.Vector
	OutStates  []State
}

func (b *batcherBlockResult) Outputs() []linalg.Vector {
	return b.OutVecs
}

func (b *batcherBlockResult) States() []State {
	return b.OutStates
}

func (b *batcherBlockResult) PropagateGradient(upstream []linalg.Vector, s []StateGrad,
	g autofunc.Gradient) []StateGrad {
	var joinedUpstream linalg.Vector
	for i, outVec := range b.OutVecs {
		if upstream != nil {
			joinedUpstream = append(joinedUpstream, upstream[i]...)
		} else {
			zeroUpstream := make(linalg.Vector, len(outVec)-b.StateSize)
			joinedUpstream = append(joinedUpstream, zeroUpstream...)
		}
		if s != nil && s[i] != nil {
			joinedUpstream = append(joinedUpstream, s[i].(VecStateGrad)...)
		} else {
			zeroUpstream := make(linalg.Vector, b.StateSize)
			joinedUpstream = append(joinedUpstream, zeroUpstream...)
		}
	}

	g[b.InPool] = make(linalg.Vector, len(b.InPool.Vector))
	b.BatcherOut.PropagateGradient(joinedUpstream, g)

	joined := g[b.InPool]
	delete(g, b.InPool)

	downIns, downStates := splitBatcherOuts(len(b.OutVecs), b.StateSize, joined)
	for i, in := range b.Ins {
		if !in.Constant(g) {
			in.PropagateGradient(downIns[i], g)
		}
	}

	var res []StateGrad
	for _, x := range downStates {
		res = append(res, VecStateGrad(x))
	}
	return res
}

func splitBatcherOuts(n, stateSize int, joined linalg.Vector) (ins, states []linalg.Vector) {
	subSize := len(joined) / n
	for i := 0; i < n; i++ {
		sub := joined[i*subSize : (i+1)*subSize]
		ins = append(ins, sub[:len(sub)-stateSize])
		states = append(states, sub[stateSize:])
	}
	return
}
