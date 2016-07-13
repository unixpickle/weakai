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

func TestNetworkSeqFuncGradients(t *testing.T) {
	rand.Seed(123)
	networkSeqFuncTestNet.Randomize()
	seqFunc := &rnn.NetworkSeqFunc{Network: networkSeqFuncTestNet}

	rSeqs := seqsToRVarSeqs(rnnSeqFuncTests)
	var nonRSeqs [][]autofunc.Result

	var params []*autofunc.Variable
	params = append(params, seqFunc.Parameters()...)
	for _, x := range rSeqs {
		var nonRSeq []autofunc.Result
		for _, y := range x {
			z := y.(*autofunc.RVariable)
			params = append(params, z.Variable)
			nonRSeq = append(nonRSeq, z.Variable)
		}
		nonRSeqs = append(nonRSeqs, nonRSeq)
	}

	rVec := autofunc.RVector{}
	for _, v := range params {
		rVec[v] = make(linalg.Vector, len(v.Vector))
		for i := range rVec[v] {
			rVec[v][i] = rand.NormFloat64()
		}
	}

	upstream := randomUpstreamSeqs(rnnSeqFuncTests, 2)
	upstreamR := randomUpstreamSeqs(rnnSeqFuncTests, 2)

	actual := autofunc.NewGradient(params)
	seqFunc.BatchSeqs(nonRSeqs).Gradient(upstream, actual)

	expected := autofunc.NewGradient(params)
	expectedR := autofunc.NewRGradient(params)
	evaluateNetworkSeqGrads(rVec, networkSeqFuncTestNet, rSeqs, upstream, upstreamR,
		expectedR, expected)

	testGradMapsEqual(t, "gradient", actual, expected)

	rOut := seqFunc.BatchSeqsR(rVec, rSeqs)
	actual.Zero()
	actualR := autofunc.NewRGradient(params)
	rOut.RGradient(upstream, upstreamR, actualR, actual)

	testGradMapsEqual(t, "gradient (r)", actual, expected)
	testGradMapsEqual(t, "r-gradient", actualR, expectedR)
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

func evaluateNetworkSeqGrads(rv autofunc.RVector, n neuralnet.Network,
	inSeqs [][]autofunc.RResult, upstream, upstreamR [][]linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	for i, inSeq := range inSeqs {
		for j, x := range inSeq {
			res := n.ApplyR(rv, x)
			upstreamCopy := make(linalg.Vector, len(upstream[i][j]))
			copy(upstreamCopy, upstream[i][j])
			rUpstreamCopy := make(linalg.Vector, len(upstreamR[i][j]))
			copy(rUpstreamCopy, upstreamR[i][j])
			res.PropagateRGradient(upstreamCopy, rUpstreamCopy, rg, g)
		}
	}
}
