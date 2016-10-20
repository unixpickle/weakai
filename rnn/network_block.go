package rnn

import (
	"encoding/json"
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var n NetworkBlock
	serializer.RegisterTypedDeserializer(n.SerializerType(), DeserializeNetworkBlock)
}

// NetworkBlock is a Block that wraps a neuralnet.Network.
// Unlike a BatcherBlock, a NetworkBlock can return a list
// of learnable parameters and can serialize itself.
type NetworkBlock struct {
	network      neuralnet.Network
	batcherBlock *BatcherBlock
}

// NewNetworkBlock creates a NetworkBlock.
func NewNetworkBlock(n neuralnet.Network, stateSize int) *NetworkBlock {
	return &NetworkBlock{
		batcherBlock: &BatcherBlock{
			B:         n.BatchLearner(),
			StateSize: stateSize,
			Start:     &autofunc.Variable{Vector: make(linalg.Vector, stateSize)},
		},
		network: n,
	}
}

// DeserializeNetworkBlock deserializes a NetworkBlock.
func DeserializeNetworkBlock(d []byte) (*NetworkBlock, error) {
	list, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	} else if len(list) != 3 {
		return nil, errors.New("bad network list length")
	}
	stateSize, ok := list[0].(serializer.Int)
	network, ok1 := list[1].(neuralnet.Network)
	initData, ok2 := list[2].(serializer.Bytes)
	if !ok || !ok1 || !ok2 {
		return nil, errors.New("bad types in network list")
	}
	var initState autofunc.Variable
	if err := json.Unmarshal(initData, &initState); err != nil {
		return nil, err
	}
	res := NewNetworkBlock(network, int(stateSize))
	res.batcherBlock.Start = &initState
	return res, nil
}

// Network returns the wrapped network.
func (n *NetworkBlock) Network() neuralnet.Network {
	return n.network
}

// StartState returns the initial state.
func (n *NetworkBlock) StartState() State {
	return n.batcherBlock.StartState()
}

// StartRState returns the initial state.
func (n *NetworkBlock) StartRState(rv autofunc.RVector) RState {
	return n.batcherBlock.StartRState(rv)
}

// PropagateStart propagates through the start state.
func (n *NetworkBlock) PropagateStart(s []State, u []StateGrad, g autofunc.Gradient) {
	n.batcherBlock.PropagateStart(s, u, g)
}

// PropagateStartR propagates through the start state.
func (n *NetworkBlock) PropagateStartR(s []RState, u []RStateGrad, rg autofunc.RGradient,
	g autofunc.Gradient) {
	n.batcherBlock.PropagateStartR(s, u, rg, g)
}

// ApplyBlock applies the block to an input.
func (n *NetworkBlock) ApplyBlock(s []State, in []autofunc.Result) BlockResult {
	return n.batcherBlock.ApplyBlock(s, in)
}

// ApplyBlockR applies the block to an input.
func (n *NetworkBlock) ApplyBlockR(rv autofunc.RVector, s []RState,
	in []autofunc.RResult) BlockRResult {
	return n.batcherBlock.ApplyBlockR(rv, s, in)
}

// Parameters returns the a slice first containing the
// initial bias variable, then containing the parameters
// of the underlying network.
func (n *NetworkBlock) Parameters() []*autofunc.Variable {
	return append([]*autofunc.Variable{n.batcherBlock.Start},
		n.network.Parameters()...)
}

// Serialize serializes the block.
func (n *NetworkBlock) Serialize() ([]byte, error) {
	initData, err := json.Marshal(n.batcherBlock.Start)
	if err != nil {
		return nil, err
	}
	serializers := []serializer.Serializer{
		serializer.Int(n.batcherBlock.StateSize),
		n.network,
		serializer.Bytes(initData),
	}
	return serializer.SerializeSlice(serializers)
}

// SerializerType returns the unique ID used to serialize
// a NetworkBlock with the serializer package.
func (n *NetworkBlock) SerializerType() string {
	return "github.com/unixpickle/weakai/rnn.NetworkBlock"
}
