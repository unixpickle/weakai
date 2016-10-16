package rnntest

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

var blockSeqFuncTests = [][]linalg.Vector{
	{{1, 2, 3}, {-1, -2, 1}, {1, 1, -1}},
	{{1, 0, 3}, {-1, -2, 1}, {1, 1, -1}},
	{{1, 2, 0}},
	{},
	{{1, 2, 3}, {-1, -2, 1}, {1, 1, -1}, {0, 0, 0}},
	{{1, 0, 3}, {-1, 1, 1}, {1, 1, -1}, {0, 0, 0}},
}

func TestBlockSeqFuncOutputs(t *testing.T) {
	rand.Seed(123)
	block := rnn.NewLSTM(3, 2)
	seqFunc := &rnn.BlockSeqFunc{Block: block}
	runner := &rnn.Runner{Block: block}

	actual := seqFunc.BatchSeqs(seqsToVarSeqs(blockSeqFuncTests)).OutputSeqs()
	expected := runner.RunAll(blockSeqFuncTests)

	testSequencesEqual(t, "outputs", actual, expected)

	actual = seqFunc.BatchSeqsR(autofunc.RVector{},
		seqsToRVarSeqs(blockSeqFuncTests)).OutputSeqs()

	testSequencesEqual(t, "outputs (r)", actual, expected)
}

func TestBlockSeqFuncROutputs(t *testing.T) {
	rand.Seed(123)
	block := rnn.NewLSTM(3, 2)
	seqFunc := &rnn.BlockSeqFunc{Block: block}

	rInputs := seqsToRVarSeqs(blockSeqFuncTests)
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
		expected = append(expected, evaluateBlockSeqFuncROutputs(rVec, block, inSeq))
	}

	testSequencesEqual(t, "r-outputs", actual, expected)
}

func TestBlockSeqFuncGradients(t *testing.T) {
	rand.Seed(123)
	block := rnn.NewLSTM(3, 2)
	seqFunc := &rnn.BlockSeqFunc{Block: block}

	parameters := make([]*autofunc.Variable, len(block.Parameters()))
	copy(parameters, block.Parameters())

	inputsR := seqsToRVarSeqs(blockSeqFuncTests)
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

	upstream, upstreamR := randomUpstreamSeqs(blockSeqFuncTests, 2),
		randomUpstreamSeqs(blockSeqFuncTests, 2)

	for i, inSeq := range inputsR {
		evaluateBlockSeqFuncGradients(autofunc.RVector{}, block, inSeq, upstream[i],
			upstreamR[i], autofunc.RGradient{}, expectedGrad)
	}

	output := seqFunc.BatchSeqs(inputs)
	output.Gradient(upstream, actualGrad)

	testGradMapsEqual(t, "gradient", actualGrad, expectedGrad)
}

func TestBlockSeqFuncRGradients(t *testing.T) {
	rand.Seed(123)
	block := rnn.NewLSTM(3, 2)
	seqFunc := &rnn.BlockSeqFunc{Block: block}

	parameters := make([]*autofunc.Variable, len(block.Parameters()))
	copy(parameters, block.Parameters())

	inputsR := seqsToRVarSeqs(blockSeqFuncTests)
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

	upstream, upstreamR := randomUpstreamSeqs(blockSeqFuncTests, 2),
		randomUpstreamSeqs(blockSeqFuncTests, 2)

	for i, inSeq := range inputsR {
		evaluateBlockSeqFuncGradients(rVector, block, inSeq, upstream[i], upstreamR[i],
			expectedRGrad, expectedGrad)
	}

	output := seqFunc.BatchSeqsR(rVector, inputsR)
	output.RGradient(upstream, upstreamR, actualRGrad, actualGrad)

	testGradMapsEqual(t, "gradient", actualGrad, expectedGrad)
	testGradMapsEqual(t, "r-gradient", actualRGrad, expectedRGrad)
}

func evaluateBlockSeqFuncROutputs(rv autofunc.RVector, b rnn.Block,
	s []autofunc.RResult) []linalg.Vector {
	res := make([]linalg.Vector, 0, len(s))
	initState := b.StartStateR(rv)
	lastState := initState.Output()
	lastRState := initState.ROutput()
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

func evaluateBlockSeqFuncGradients(rv autofunc.RVector, b rnn.Block,
	inputs []autofunc.RResult, upstream,
	upstreamR []linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	var blockOuts []rnn.BlockROutput
	var inStates []*autofunc.RVariable
	initState := b.StartStateR(rv)
	lastState := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: initState.Output()},
		ROutputVec: initState.ROutput(),
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
	initState.PropagateRGradient(upstreamState, upstreamStateR, rg, g)
}
