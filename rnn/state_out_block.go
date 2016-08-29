package rnn

import (
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
)

// StateOutBlock is a Block which wraps another Block and
// outputs that Block's state.
// As part of this process, the wrapped Block's output
// is discarded.
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

// StateSize returns the state size of the wrapped Block.
func (s *StateOutBlock) StateSize() int {
	return s.StateSize()
}

// StartState returns the wrapped Block's start state.
func (s *StateOutBlock) StartState() autofunc.Result {
	return s.Block.StartState()
}

// StartStateR returns the wrapped Block's start state.
func (s *StateOutBlock) StartStateR(rv autofunc.RVector) autofunc.RResult {
	return s.Block.StartStateR(rv)
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

// Batch applies the wrapped block to the input, yielding
// an output with the wrapped block's state used as the
// block's output.
func (s *StateOutBlock) Batch(in *BlockInput) BlockOutput {
	return &stateOutBlockOutput{
		WrappedOut: s.Block.Batch(in),
	}
}

// BatchR is like Batch, but with r-operator support.
func (s *StateOutBlock) BatchR(rv autofunc.RVector, in *BlockRInput) BlockROutput {
	return &stateOutBlockROutput{
		WrappedOut: s.Block.BatchR(rv, in),
	}
}

// SerializerType returns the unique ID used to serialize
// StateOutBlocks with the serializer package.
func (s *StateOutBlock) SerializerType() string {
	return serializerTypeStateOutBlock
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

type stateOutBlockOutput struct {
	WrappedOut BlockOutput
}

func (s *stateOutBlockOutput) States() []linalg.Vector {
	return s.WrappedOut.States()
}

func (s *stateOutBlockOutput) Outputs() []linalg.Vector {
	return s.WrappedOut.States()
}

func (s *stateOutBlockOutput) Gradient(u *UpstreamGradient, g autofunc.Gradient) {
	innerUpstream := &UpstreamGradient{
		States:  make([]linalg.Vector, len(s.WrappedOut.States())),
		Outputs: make([]linalg.Vector, len(s.WrappedOut.Outputs())),
	}

	var zeroUpstream linalg.Vector
	for i, x := range s.WrappedOut.Outputs() {
		if i == 0 {
			zeroUpstream = make(linalg.Vector, len(x))
		}
		innerUpstream.Outputs[i] = zeroUpstream
	}

	if u.States != nil {
		for i, x := range u.States {
			innerUpstream.States[i] = x
		}
	}
	if u.Outputs != nil {
		for i, x := range u.Outputs {
			if u.States != nil {
				innerUpstream.States[i] = x.Copy().Add(innerUpstream.States[i])
			} else {
				innerUpstream.States[i] = x
			}
		}
	}

	s.WrappedOut.Gradient(innerUpstream, g)
}

type stateOutBlockROutput struct {
	WrappedOut BlockROutput
}

func (s *stateOutBlockROutput) States() []linalg.Vector {
	return s.WrappedOut.States()
}

func (s *stateOutBlockROutput) Outputs() []linalg.Vector {
	return s.WrappedOut.States()
}

func (s *stateOutBlockROutput) RStates() []linalg.Vector {
	return s.WrappedOut.RStates()
}

func (s *stateOutBlockROutput) ROutputs() []linalg.Vector {
	return s.WrappedOut.RStates()
}

func (s *stateOutBlockROutput) RGradient(u *UpstreamRGradient, rg autofunc.RGradient,
	g autofunc.Gradient) {
	innerUpstream := &UpstreamRGradient{
		UpstreamGradient: UpstreamGradient{
			States:  make([]linalg.Vector, len(s.WrappedOut.States())),
			Outputs: make([]linalg.Vector, len(s.WrappedOut.Outputs())),
		},
		RStates:  make([]linalg.Vector, len(s.WrappedOut.States())),
		ROutputs: make([]linalg.Vector, len(s.WrappedOut.Outputs())),
	}

	var zeroUpstream linalg.Vector
	for i, x := range s.WrappedOut.Outputs() {
		if i == 0 {
			zeroUpstream = make(linalg.Vector, len(x))
		}
		innerUpstream.Outputs[i] = zeroUpstream
		innerUpstream.ROutputs[i] = zeroUpstream
	}

	if u.States != nil {
		for i, x := range u.States {
			innerUpstream.States[i] = x
			innerUpstream.RStates[i] = u.RStates[i]
		}
	}
	if u.Outputs != nil {
		for i, x := range u.Outputs {
			xR := u.ROutputs[i]
			if u.States != nil {
				innerUpstream.States[i] = x.Copy().Add(innerUpstream.States[i])
				innerUpstream.RStates[i] = xR.Copy().Add(innerUpstream.RStates[i])
			} else {
				innerUpstream.States[i] = x
				innerUpstream.RStates[i] = xR
			}
		}
	}

	s.WrappedOut.RGradient(innerUpstream, rg, g)
}
