package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/weakai/neuralnet"
)

// A NetworkSeqFunc is a seqfunc.RFunc which applies a
// neuralnet.Network to each input to generate an output.
type NetworkSeqFunc struct {
	Network neuralnet.Network
}

// DeserializeNetworkSeqFunc deserializes a NetworkSeqFunc
// that was previously serialized.
func DeserializeNetworkSeqFunc(d []byte) (*NetworkSeqFunc, error) {
	net, err := neuralnet.DeserializeNetwork(d)
	if err != nil {
		return nil, err
	}
	return &NetworkSeqFunc{Network: net}, nil
}

// ApplySeqs applies the network to the sequences.
func (n *NetworkSeqFunc) ApplySeqs(in seqfunc.Result) seqfunc.Result {
	mb := &seqfunc.MapBatcher{B: n.Network.BatchLearner()}
	return mb.ApplySeqs(in)
}

// ApplySeqsR applies the network to the sequences.
func (n *NetworkSeqFunc) ApplySeqsR(rv autofunc.RVector, in seqfunc.RResult) seqfunc.RResult {
	mb := &seqfunc.MapRBatcher{B: n.Network.BatchLearner()}
	return mb.ApplySeqsR(rv, in)
}

// Parameters returns the network's parameters.
func (n *NetworkSeqFunc) Parameters() []*autofunc.Variable {
	return n.Network.Parameters()
}

// SerializerType returns the unique ID used to serialize
// a NetworkSeqFunc with the serializer package.
func (n *NetworkSeqFunc) SerializerType() string {
	return "github.com/unixpickle/weakai/rnn.NetworkSeqFunc"
}

// Serialize serializes the NetworkSeqFunc.
func (n *NetworkSeqFunc) Serialize() ([]byte, error) {
	return n.Network.Serialize()
}
