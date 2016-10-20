package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A State represents some internal, recurrent state.
// For instance, it might represent the current contents
// of LSTM blocks.
type State interface{}

// A StateGrad represents the gradient of a loss value
// with respect to a given State.
type StateGrad interface{}

// An RState is like a State, but with information about
// second derivatives with respect to some variable R.
type RState interface{}

// An RStateGrad is like a StateGrad, but for an RState.
type RStateGrad interface{}

// A VecState is a State which stores a vector.
type VecState linalg.Vector

// A VecStateGrad is a StateGrad which stores a vector.
type VecStateGrad linalg.Vector

// A VecRState is an RState which stores a vector and its
// derivative.
type VecRState struct {
	State  linalg.Vector
	RState linalg.Vector
}

// A VecRStateGrad is a StateGrad which stores a vector
// and its derivative.
type VecRStateGrad struct {
	State  linalg.Vector
	RState linalg.Vector
}

// PropagateVarState is a helper to propagate a gradient
// through a VecState that was derived from a variable.
func PropagateVarState(v *autofunc.Variable, s []StateGrad, g autofunc.Gradient) {
	if vec, ok := g[v]; ok {
		for _, x := range s {
			vec.Add(linalg.Vector(x.(VecStateGrad)))
		}
	}
}

// PropagateVarStateR is like PropagateVarState, but with
// support for the r-operator.
func PropagateVarStateR(v *autofunc.Variable, s []RStateGrad, rg autofunc.RGradient,
	g autofunc.Gradient) {
	if g != nil {
		if vec, ok := g[v]; ok {
			for _, x := range s {
				vec.Add(x.(VecRStateGrad).State)
			}
		}
	}
	if vec, ok := rg[v]; ok {
		for _, x := range s {
			vec.Add(x.(VecRStateGrad).RState)
		}
	}
}

// PoolVecStates creates a pool variable for each VecState
// in a list of VecStates.
// It also puts the same variables in a slice of
// autofunc.Results for convenience.
func PoolVecStates(s []State) ([]*autofunc.Variable, []autofunc.Result) {
	vars := make([]*autofunc.Variable, len(s))
	reses := make([]autofunc.Result, len(s))
	for i, x := range s {
		vars[i] = &autofunc.Variable{Vector: linalg.Vector(x.(VecState))}
		reses[i] = vars[i]
	}
	return vars, reses
}

// PoolVecRStates is like PoolVecStates but for VecRStates
// rather than VecStates.
func PoolVecRStates(s []RState) ([]*autofunc.Variable, []autofunc.RResult) {
	vars := make([]*autofunc.Variable, len(s))
	reses := make([]autofunc.RResult, len(s))
	for i, x := range s {
		vState := x.(VecRState)
		vars[i] = &autofunc.Variable{Vector: vState.State}
		reses[i] = &autofunc.RVariable{
			Variable:   vars[i],
			ROutputVec: vState.RState,
		}
	}
	return vars, reses
}

// PropagateVecStatePool calls f while the gradient g is
// configured to capture the gradients of the pool
// variables.
// The resulting pool gradients are returned as a slice
// of VecStateGrads cast to StateGrads.
func PropagateVecStatePool(g autofunc.Gradient, pool []*autofunc.Variable, f func()) []StateGrad {
	res := make([]StateGrad, len(pool))
	for _, v := range pool {
		g[v] = make(linalg.Vector, len(v.Vector))
	}
	f()
	for i, v := range pool {
		res[i] = VecStateGrad(g[v])
		delete(g, v)
	}
	return res
}

// PropagateVecRStatePool is like PropagateVecStatePool
// but for pooled VecRStates.
//
// The g argument may be nil.
func PropagateVecRStatePool(rg autofunc.RGradient, g autofunc.Gradient,
	pool []*autofunc.Variable, f func()) []RStateGrad {
	res := make([]RStateGrad, len(pool))
	if g == nil {
		g = autofunc.Gradient{}
	}
	for _, v := range pool {
		g[v] = make(linalg.Vector, len(v.Vector))
		rg[v] = make(linalg.Vector, len(v.Vector))
	}
	f()
	for i, v := range pool {
		res[i] = VecRStateGrad{
			State:  g[v],
			RState: rg[v],
		}
		delete(g, v)
		delete(rg, v)
	}
	return res
}
