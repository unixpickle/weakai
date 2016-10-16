package rnn

import (
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
)

func init() {
	var s StateOutBlock
	serializer.RegisterTypedDeserializer(s.SerializerType(), DeserializeStateOutBlock)
}

// A StateOutBlock wraps another Block and uses the
// block's state as output.
// The output from the wrapped Block is discarded.
//
// The wrapped Block must use VecState and VecRState.
type StateOutBlock struct {
	Block Block
}

// DeserializeStateOutBlock deserializes a StateOutBlock.
func DeserializeStateOutBlock(d []byte) (*StateOutBlock, error) {
	b, err := serializer.DeserializeWithType(d)
	if err != nil {
		return nil, err
	}
	block, ok := b.(Block)
	if !ok {
		return nil, fmt.Errorf("wrapped object is not a Block: %T", b)
	}
	return &StateOutBlock{Block: block}, nil
}

// StartState returns the wrapped Block's start state.
func (s *StateOutBlock) StartState() State {
	return s.Block.StartState()
}

// StartStateR returns the wrapped Block's start state.
func (s *StateOutBlock) StartRState(rv autofunc.RVector) RState {
	return s.Block.StartRState(rv)
}

// PropagateStart propagates through the start state.
func (s *StateOutBlock) PropagateStart(u []StateGrad, g autofunc.Gradient) {
	s.Block.PropagateStart(u, g)
}

// PropagateStartR propagates through the start state.
func (s *StateOutBlock) PropagateStartR(u []RStateGrad, rg autofunc.RGradient,
	g autofunc.Gradient) {
	s.Block.PropagateStartR(u, rg, g)
}

// ApplyBlock applies the block to an input.
func (s *StateOutBlock) ApplyBlock(state []State, in []autofunc.Result) BlockResult {
	return &stateOutBlockResult{
		WrappedOut: s.Block.ApplyBlock(state, in),
	}
}

// ApplyBlockR applies the block to an input.
func (s *StateOutBlock) ApplyBlockR(rv autofunc.RVector, state []RState,
	in []autofunc.RResult) BlockRResult {
	return &stateOutBlockRResult{
		WrappedOut: s.Block.ApplyBlockR(rv, state, in),
	}
}

// Parameters returns the parameters of the wrapped Block,
// or nil if the Block is not an sgd.Learner.
func (s *StateOutBlock) Parameters() []*autofunc.Variable {
	ler, ok := s.Block.(sgd.Learner)
	if !ok {
		return nil
	}
	return ler.Parameters()
}

// SerializerType returns the unique ID used to serialize
// StateOutBlocks with the serializer package.
func (s *StateOutBlock) SerializerType() string {
	return "github.com/unixpickle/weakai/rnn.StateOutBlock"
}

// Serialize attempts to serialize this block by
// serializing the wrapped block.
// If the wrapped block is not a serializer.Serializer,
// this will fail.
func (s *StateOutBlock) Serialize() ([]byte, error) {
	ser, ok := s.Block.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("type is not a Serializer: %T", s.Block)
	}
	return serializer.SerializeWithType(ser)
}

type stateOutBlockResult struct {
	WrappedOut BlockResult
}

func (s *stateOutBlockResult) States() []State {
	return s.WrappedOut.States()
}

func (s *stateOutBlockResult) Outputs() []linalg.Vector {
	res := make([]linalg.Vector, len(s.WrappedOut.States()))
	for i, state := range s.WrappedOut.States() {
		res[i] = linalg.Vector(state.(VecState))
	}
	return res
}

func (s *stateOutBlockResult) PropagateGradient(u []linalg.Vector, su []StateGrad,
	g autofunc.Gradient) []StateGrad {
	downstream := make([]StateGrad, len(s.WrappedOut.Outputs()))
	for i := range s.WrappedOut.Outputs() {
		var vec linalg.Vector
		if u != nil {
			vec = u[i].Copy()
		}
		if su != nil && su[i] != nil {
			sVec := su[i].(VecStateGrad)
			if vec == nil {
				vec = linalg.Vector(sVec).Copy()
			} else {
				vec.Add(linalg.Vector(sVec))
			}
		}
		if vec != nil {
			downstream[i] = VecStateGrad(vec)
		}
	}
	return s.WrappedOut.PropagateGradient(nil, downstream, g)
}

type stateOutBlockRResult struct {
	WrappedOut BlockRResult
}

func (s *stateOutBlockRResult) RStates() []RState {
	return s.WrappedOut.RStates()
}

func (s *stateOutBlockRResult) Outputs() []linalg.Vector {
	res := make([]linalg.Vector, len(s.WrappedOut.RStates()))
	for i, state := range s.WrappedOut.RStates() {
		res[i] = linalg.Vector(state.(VecRState).State)
	}
	return res
}

func (s *stateOutBlockRResult) ROutputs() []linalg.Vector {
	res := make([]linalg.Vector, len(s.WrappedOut.RStates()))
	for i, state := range s.WrappedOut.RStates() {
		res[i] = linalg.Vector(state.(VecRState).RState)
	}
	return res
}

func (s *stateOutBlockRResult) PropagateRGradient(u, uR []linalg.Vector, su []RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) []RStateGrad {
	downstream := make([]RStateGrad, len(s.WrappedOut.Outputs()))
	for i := range s.WrappedOut.Outputs() {
		var vec, vecR linalg.Vector
		if u != nil {
			vec = u[i].Copy()
			vecR = uR[i].Copy()
		}
		if su != nil && su[i] != nil {
			sVec := su[i].(VecRStateGrad)
			if vec == nil {
				vec = sVec.State.Copy()
				vecR = sVec.RState.Copy()
			} else {
				vec.Add(sVec.State)
				vecR.Add(sVec.RState)
			}
		}
		if vec != nil {
			downstream[i] = VecRStateGrad{State: vec, RState: vecR}
		}
	}
	return s.WrappedOut.PropagateRGradient(nil, nil, downstream, rg, g)
}
