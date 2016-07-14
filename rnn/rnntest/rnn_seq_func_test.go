package rnntest

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

var rnnSeqFuncTests = [][]linalg.Vector{
	{{1, 2, 3}, {-1, -2, 1}, {1, 1, -1}},
	{{1, 0, 3}, {-1, -2, 1}, {1, 1, -1}},
	{{1, 2, 0}},
	{},
	{{1, 2, 3}, {-1, -2, 1}, {1, 1, -1}, {0, 0, 0}},
	{{1, 0, 3}, {-1, 1, 1}, {1, 1, -1}, {0, 0, 0}},
}

func TestRNNSeqFuncOutputs(t *testing.T) {
	rand.Seed(123)
	block := rnn.NewLSTM(3, 2)
	seqFunc := &rnn.RNNSeqFunc{Block: block}
	runner := &rnn.Runner{Block: block}

	actual := seqFunc.BatchSeqs(seqsToVarSeqs(rnnSeqFuncTests)).OutputSeqs()
	expected := runner.RunAll(rnnSeqFuncTests)

	testSequencesEqual(t, "outputs", actual, expected)

	actual = seqFunc.BatchSeqsR(autofunc.RVector{},
		seqsToRVarSeqs(rnnSeqFuncTests)).OutputSeqs()

	testSequencesEqual(t, "outputs (r)", actual, expected)
}

func TestRNNSeqFuncROutputs(t *testing.T) {
	rand.Seed(123)
	block := rnn.NewLSTM(3, 2)
	seqFunc := &rnn.RNNSeqFunc{Block: block}

	rInputs := seqsToRVarSeqs(rnnSeqFuncTests)
	rVec := autofunc.RVector{}
	for _, v := range block.Parameters() {
		rVec[v] = make(linalg.Vector, len(v.Vector))
		for i := range rVec[v] {
			rVec[v][i] = rand.NormFloat64()
		}
	}

	actual := seqFunc.BatchSeqsR(rVec, rInputs).ROutputSeqs()
	var expected [][]linalg.Vector
	for _, inSeq := range rInputs {
		expected = append(expected, evaluateRNNSeqFuncROutputs(rVec, block, inSeq))
	}

	testSequencesEqual(t, "r-outputs", actual, expected)
}

func TestRNNSeqFuncGradients(t *testing.T) {
	rand.Seed(123)
	block := rnn.NewLSTM(3, 2)
	seqFunc := &rnn.RNNSeqFunc{Block: block}

	parameters := make([]*autofunc.Variable, len(block.Parameters()))
	copy(parameters, block.Parameters())

	inputsR := seqsToRVarSeqs(rnnSeqFuncTests)
	var inputs [][]autofunc.Result
	for _, v := range inputsR {
		var list []autofunc.Result
		for _, in := range v {
			rv := in.(*autofunc.RVariable)
			parameters = append(parameters, rv.Variable)
			list = append(list, rv.Variable)
		}
		inputs = append(inputs, list)
	}

	actualGrad := autofunc.NewGradient(parameters)
	expectedGrad := autofunc.NewGradient(parameters)

	upstream, upstreamR := randomUpstreamSeqs(rnnSeqFuncTests, 2),
		randomUpstreamSeqs(rnnSeqFuncTests, 2)

	for i, inSeq := range inputsR {
		evaluateRNNSeqFuncGradients(autofunc.RVector{}, block, inSeq, upstream[i],
			upstreamR[i], autofunc.RGradient{}, expectedGrad)
	}

	output := seqFunc.BatchSeqs(inputs)
	output.Gradient(upstream, actualGrad)

	testGradMapsEqual(t, "gradient", actualGrad, expectedGrad)
}

func TestRNNSeqFuncRGradients(t *testing.T) {
	rand.Seed(123)
	block := rnn.NewLSTM(3, 2)
	seqFunc := &rnn.RNNSeqFunc{Block: block}

	parameters := make([]*autofunc.Variable, len(block.Parameters()))
	copy(parameters, block.Parameters())

	inputsR := seqsToRVarSeqs(rnnSeqFuncTests)
	for _, v := range inputsR {
		for _, in := range v {
			rv := in.(*autofunc.RVariable)
			parameters = append(parameters, rv.Variable)
		}
	}

	rVector := autofunc.RVector{}
	for _, p := range parameters {
		rVector[p] = make(linalg.Vector, len(p.Vector))
		for i := range rVector[p] {
			rVector[p][i] = rand.NormFloat64()
		}
	}

	actualGrad := autofunc.NewGradient(parameters)
	actualRGrad := autofunc.NewRGradient(parameters)
	expectedGrad := autofunc.NewGradient(parameters)
	expectedRGrad := autofunc.NewRGradient(parameters)

	upstream, upstreamR := randomUpstreamSeqs(rnnSeqFuncTests, 2),
		randomUpstreamSeqs(rnnSeqFuncTests, 2)

	for i, inSeq := range inputsR {
		evaluateRNNSeqFuncGradients(rVector, block, inSeq, upstream[i], upstreamR[i],
			expectedRGrad, expectedGrad)
	}

	output := seqFunc.BatchSeqsR(rVector, inputsR)
	output.RGradient(upstream, upstreamR, actualRGrad, actualGrad)

	testGradMapsEqual(t, "gradient", actualGrad, expectedGrad)
	testGradMapsEqual(t, "r-gradient", actualRGrad, expectedRGrad)
}

func evaluateRNNSeqFuncROutputs(rv autofunc.RVector, b rnn.Block,
	s []autofunc.RResult) []linalg.Vector {
	res := make([]linalg.Vector, 0, len(s))
	lastState := make(linalg.Vector, b.StateSize())
	lastRState := make(linalg.Vector, b.StateSize())
	for _, in := range s {
		inVar := &autofunc.RVariable{
			Variable:   &autofunc.Variable{Vector: in.Output()},
			ROutputVec: in.ROutput(),
		}
		inState := &autofunc.RVariable{
			Variable:   &autofunc.Variable{Vector: lastState},
			ROutputVec: lastRState,
		}
		out := b.BatchR(rv, &rnn.BlockRInput{
			Inputs: []*autofunc.RVariable{inVar},
			States: []*autofunc.RVariable{inState},
		})
		res = append(res, out.ROutputs()[0])
		lastState = out.States()[0]
		lastRState = out.RStates()[0]
	}
	return res
}

func evaluateRNNSeqFuncGradients(rv autofunc.RVector, b rnn.Block,
	inputs []autofunc.RResult, upstream,
	upstreamR []linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	var blockOuts []rnn.BlockROutput
	var inStates []*autofunc.RVariable
	lastState := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: make(linalg.Vector, b.StateSize())},
		ROutputVec: make(linalg.Vector, b.StateSize()),
	}

	// Perform the forward pass.
	for _, in := range inputs {
		inVar := in.(*autofunc.RVariable)
		out := b.BatchR(rv, &rnn.BlockRInput{
			Inputs: []*autofunc.RVariable{inVar},
			States: []*autofunc.RVariable{lastState},
		})
		inStates = append(inStates, lastState)
		lastState = &autofunc.RVariable{
			Variable:   &autofunc.Variable{Vector: out.States()[0]},
			ROutputVec: out.RStates()[0],
		}
		blockOuts = append(blockOuts, out)
	}

	upstreamState := make(linalg.Vector, b.StateSize())
	upstreamStateR := make(linalg.Vector, b.StateSize())

	// Perform the backward pass.
	for i := len(blockOuts) - 1; i >= 0; i-- {
		out := blockOuts[i]
		stateVar := inStates[i].Variable
		g[stateVar] = make(linalg.Vector, b.StateSize())
		rg[stateVar] = make(linalg.Vector, b.StateSize())
		out.RGradient(&rnn.UpstreamRGradient{
			UpstreamGradient: rnn.UpstreamGradient{
				States:  []linalg.Vector{upstreamState},
				Outputs: []linalg.Vector{upstream[i]},
			},
			RStates:  []linalg.Vector{upstreamStateR},
			ROutputs: []linalg.Vector{upstreamR[i]},
		}, rg, g)
		upstreamState = g[stateVar]
		upstreamStateR = rg[stateVar]
		delete(g, stateVar)
		delete(rg, stateVar)
	}
}
