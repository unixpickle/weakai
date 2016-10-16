package rnn

import "github.com/unixpickle/num-analysis/linalg"

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
