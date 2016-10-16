package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/weakai/neuralnet"
)

// A NetworkSeqFunc is a SeqFunc which applies a
// neuralnet.Network to each input to generate an
// output.
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

func (n *NetworkSeqFunc) ApplySeqs(in seqfunc.Result) seqfunc.Result {
	mb := &seqfunc.MapBatcher{B: n.Network.BatchLearner()}
	return mb.ApplySeqs(in)
}

func (n *NetworkSeqFunc) BatchSeqsR(rv autofunc.RVector, seqs [][]autofunc.RResult) RResultSeqs {
	mb := &seqfunc.MapRBatcher{B: n.Network.BatchLearner()}
	return mb.ApplySeqsR(rv, seqs)
}

func (n *NetworkSeqFunc) Parameters() []*autofunc.Variable {
	return n.Network.Parameters()
}

func (n *NetworkSeqFunc) SerializerType() string {
	return serializerTypeNetworkSeqFunc
}

func (n *NetworkSeqFunc) Serialize() ([]byte, error) {
	return n.Network.Serialize()
}
