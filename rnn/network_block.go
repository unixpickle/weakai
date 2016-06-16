package rnn

import (
	"errors"

	"github.com/unixpickle/autofunc"
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
			F:            n.BatchLearner(),
			StateSizeVal: stateSize,
		},
		network: n,
	}
}

func DeserializeNetworkBlock(d []byte) (serializer.Serializer, error) {
	list, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	} else if len(list) != 2 {
		return nil, errors.New("bad network list length")
	}
	stateSize, ok := list[0].(serializer.Int)
	network, ok1 := list[1].(neuralnet.Network)
	if ok && ok1 {
		return nil, errors.New("bad types in network list")
	}
	return NewNetworkBlock(network, int(stateSize)), nil
}

func (n *NetworkBlock) Network() neuralnet.Network {
	return n.network
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

func (n *NetworkBlock) Parameters() []*autofunc.Variable {
	return n.network.Parameters()
}

func (n *NetworkBlock) Serialize() ([]byte, error) {
	size := serializer.Int(n.batcherBlock.StateSizeVal)
	serializers := []serializer.Serializer{size, n.network}
	return serializer.SerializeSlice(serializers)
}

func (n *NetworkBlock) SerializerType() string {
	return serializerTypeNetworkBlock
}
