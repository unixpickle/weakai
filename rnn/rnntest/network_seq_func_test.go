package rnntest

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

var networkSeqFuncTestNet = neuralnet.Network{
	&neuralnet.DenseLayer{
		InputCount:  3,
		OutputCount: 2,
	},
	&neuralnet.Sigmoid{},
}

func TestNetworkSeqFuncOutputs(t *testing.T) {
	rand.Seed(123)
	networkSeqFuncTestNet.Randomize()
	seqFunc := &rnn.NetworkSeqFunc{Network: networkSeqFuncTestNet}

	rSeqs := seqsToRVarSeqs(rnnSeqFuncTests)
	rVec := autofunc.RVector{}

	actual := seqFunc.BatchSeqs(seqsToVarSeqs(rnnSeqFuncTests)).OutputSeqs()
	expected, _ := evaluateNetworkSeqROutputs(rVec, networkSeqFuncTestNet, rSeqs)

	testSequencesEqual(t, actual, expected)

	actual = seqFunc.BatchSeqsR(autofunc.RVector{},
		seqsToRVarSeqs(rnnSeqFuncTests)).OutputSeqs()

	testSequencesEqual(t, actual, expected)
}

func TestNetworkSeqFuncROutputs(t *testing.T) {
	rand.Seed(123)
	networkSeqFuncTestNet.Randomize()
	seqFunc := &rnn.NetworkSeqFunc{Network: networkSeqFuncTestNet}

	rSeqs := seqsToRVarSeqs(rnnSeqFuncTests)
	rVec := autofunc.RVector{}
	for _, v := range seqFunc.Parameters() {
		rVec[v] = make(linalg.Vector, len(v.Vector))
		for i := range rVec[v] {
			rVec[v][i] = rand.NormFloat64()
		}
	}

	_, expectedR := evaluateNetworkSeqROutputs(rVec, networkSeqFuncTestNet, rSeqs)
	actualR := seqFunc.BatchSeqsR(rVec, seqsToRVarSeqs(rnnSeqFuncTests)).ROutputSeqs()
	testSequencesEqual(t, actualR, expectedR)
}

func evaluateNetworkSeqROutputs(rv autofunc.RVector, n neuralnet.Network,
	inSeqs [][]autofunc.RResult) (output, outputR [][]linalg.Vector) {
	for _, inSeq := range inSeqs {
		var outVec, outRVec []linalg.Vector
		for _, x := range inSeq {
			res := n.ApplyR(rv, x)
			outVec = append(outVec, res.Output())
			outRVec = append(outRVec, res.ROutput())
		}
		output = append(output, outVec)
		outputR = append(outputR, outRVec)
	}
	return
}
