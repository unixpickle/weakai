package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
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
func DeserializeNetworkSeqFunc(d []byte) (serializer.Serializer, error) {
	net, err := neuralnet.DeserializeNetwork(d)
	if err != nil {
		return nil, err
	}
	return &NetworkSeqFunc{Network: net}, nil
}

func (n *NetworkSeqFunc) BatchSeqs(seqs [][]autofunc.Result) ResultSeqs {
	batcher := n.Network.BatchLearner()

	res := &networkSeqFuncResult{
		PackedOut: make([][]linalg.Vector, len(seqs)),
	}

	var t int
	for {
		var batchIn []autofunc.Result
		for _, seq := range seqs {
			if len(seq) > t {
				batchIn = append(batchIn, seq[t])
			}
		}
		if len(batchIn) == 0 {
			break
		}
		out := batcher.Batch(autofunc.Concat(batchIn...), len(batchIn))
		res.Outputs = append(res.Outputs, out)

		outVec := out.Output()
		outSize := len(outVec) / len(batchIn)
		offset := 0
		for l, seq := range seqs {
			if len(seq) > t {
				vec := outVec[offset : offset+outSize]
				offset += outSize
				res.PackedOut[l] = append(res.PackedOut[l], vec)
			}
		}

		t++
	}

	return res
}

func (n *NetworkSeqFunc) BatchSeqsR(rv autofunc.RVector, seqs [][]autofunc.RResult) RResultSeqs {
	batcher := n.Network.BatchLearner()

	res := &networkSeqFuncRResult{
		PackedOut:  make([][]linalg.Vector, len(seqs)),
		RPackedOut: make([][]linalg.Vector, len(seqs)),
	}

	var t int
	for {
		var batchIn []autofunc.RResult
		for _, seq := range seqs {
			if len(seq) > t {
				batchIn = append(batchIn, seq[t])
			}
		}
		if len(batchIn) == 0 {
			break
		}
		out := batcher.BatchR(rv, autofunc.ConcatR(batchIn...), len(batchIn))
		res.Outputs = append(res.Outputs, out)

		outVec := out.Output()
		rOutVec := out.ROutput()
		outSize := len(outVec) / len(batchIn)
		offset := 0
		for l, seq := range seqs {
			if len(seq) > t {
				vec := outVec[offset : offset+outSize]
				vecR := rOutVec[offset : offset+outSize]
				offset += outSize
				res.PackedOut[l] = append(res.PackedOut[l], vec)
				res.RPackedOut[l] = append(res.RPackedOut[l], vecR)
			}
		}

		t++
	}

	return res
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

type networkSeqFuncResult struct {
	Outputs   []autofunc.Result
	PackedOut [][]linalg.Vector
}

func (n *networkSeqFuncResult) OutputSeqs() [][]linalg.Vector {
	return n.PackedOut
}

func (n *networkSeqFuncResult) Gradient(upstream [][]linalg.Vector, g autofunc.Gradient) {
	var t int
	for {
		var upstreamVec linalg.Vector
		for _, us := range upstream {
			if len(us) > t {
				upstreamVec = append(upstreamVec, us[t]...)
			}
		}
		if len(upstreamVec) == 0 {
			break
		}
		n.Outputs[t].PropagateGradient(upstreamVec, g)
		t++
	}
}

type networkSeqFuncRResult struct {
	Outputs    []autofunc.RResult
	PackedOut  [][]linalg.Vector
	RPackedOut [][]linalg.Vector
}

func (n *networkSeqFuncRResult) OutputSeqs() [][]linalg.Vector {
	return n.PackedOut
}

func (n *networkSeqFuncRResult) ROutputSeqs() [][]linalg.Vector {
	return n.RPackedOut
}

func (n *networkSeqFuncRResult) RGradient(upstream, upstreamR [][]linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	var t int
	for {
		var upstreamVec, upstreamVecR linalg.Vector
		for l, us := range upstream {
			if len(us) > t {
				upstreamVec = append(upstreamVec, us[t]...)
				upstreamVecR = append(upstreamVecR, upstreamR[l][t]...)
			}
		}
		if len(upstreamVec) == 0 {
			break
		}
		n.Outputs[t].PropagateRGradient(upstreamVec, upstreamVecR, rg, g)
		t++
	}
}
