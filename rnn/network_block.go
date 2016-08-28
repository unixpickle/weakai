package rnn

import (
	"encoding/json"
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

// NetworkBlock is a Block that wraps a neuralnet.Network.
// Unlike a BatcherBlock, a NetworkBlock can return a list
// of learnable parameters and can serialize itself.
type NetworkBlock struct {
	network      neuralnet.Network
	batcherBlock *BatcherBlock
}

func NewNetworkBlock(n neuralnet.Network, stateSize int) *NetworkBlock {
	return &NetworkBlock{
		batcherBlock: &BatcherBlock{
			F:             n.BatchLearner(),
			StateSizeVal:  stateSize,
			StartStateVar: &autofunc.Variable{Vector: make(linalg.Vector, stateSize)},
		},
		network: n,
	}
}

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
	res.batcherBlock.StartStateVar = &initState
	return res, nil
}

func (n *NetworkBlock) Network() neuralnet.Network {
	return n.network
}

func (n *NetworkBlock) StartState() autofunc.Result {
	return n.batcherBlock.StartState()
}

func (n *NetworkBlock) StartStateR(rv autofunc.RVector) autofunc.RResult {
	return n.batcherBlock.StartStateR(rv)
}

func (n *NetworkBlock) StateSize() int {
	return n.batcherBlock.StateSize()
}

func (n *NetworkBlock) Batch(in *BlockInput) BlockOutput {
	return n.batcherBlock.Batch(in)
}

func (n *NetworkBlock) BatchR(v autofunc.RVector, in *BlockRInput) BlockROutput {
	return n.batcherBlock.BatchR(v, in)
}

// Parameters returns the a slice first containing the
// initial bias variable, then containing the parameters
// of the underlying network.
func (n *NetworkBlock) Parameters() []*autofunc.Variable {
	return append([]*autofunc.Variable{n.batcherBlock.StartStateVar},
		n.network.Parameters()...)
}

func (n *NetworkBlock) Serialize() ([]byte, error) {
	initData, err := json.Marshal(n.batcherBlock.StartStateVar)
	if err != nil {
		return nil, err
	}
	serializers := []serializer.Serializer{
		serializer.Int(n.batcherBlock.StateSizeVal),
		n.network,
		serializer.Bytes(initData),
	}
	return serializer.SerializeSlice(serializers)
}

func (n *NetworkBlock) SerializerType() string {
	return serializerTypeNetworkBlock
}
