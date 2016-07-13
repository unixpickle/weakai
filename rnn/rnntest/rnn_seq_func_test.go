package rnntest

import (
	"math"
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

const rnnSeqFuncTestPrec = 1e-5

func TestRNNSeqFuncOutputs(t *testing.T) {
	rand.Seed(123)
	block := rnn.NewLSTM(3, 2)
	seqFunc := &rnn.RNNSeqFunc{Block: block}
	runner := &rnn.Runner{Block: block}

	actual := seqFunc.BatchSeqs(seqsToVarSeqs(rnnSeqFuncTests)).OutputSeqs()
	expected := runner.RunAll(rnnSeqFuncTests)

	testSequencesEqual(t, actual, expected)

	actual = seqFunc.BatchSeqsR(autofunc.RVector{},
		seqsToRVarSeqs(rnnSeqFuncTests)).OutputSeqs()

	testSequencesEqual(t, actual, expected)
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

	testSequencesEqual(t, actual, expected)
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

func testSequencesEqual(t *testing.T, actual, expected [][]linalg.Vector) {
	if len(actual) != len(expected) {
		t.Errorf("expected %d outputs but got %d", len(expected), len(actual))
		return
	}
	for i, xs := range expected {
		as := actual[i]
		if len(xs) != len(as) {
			t.Errorf("output %d: expected %d timesteps but got %d",
				i, len(xs), len(as))
			continue
		}
		for time, xVec := range xs {
			aVec := as[time]
			if len(xVec) != len(aVec) {
				t.Errorf("output %d time %d: expected len %d got %d",
					i, time, len(xVec), len(aVec))
			} else {
				for j, x := range xVec {
					a := aVec[j]
					if math.Abs(a-x) > rnnSeqFuncTestPrec {
						t.Errorf("output %d time %d entry %d: expected %f got %f",
							i, time, j, x, a)
					}
				}
			}
		}
	}
}

func testGradMapsEqual(t *testing.T, label string, actual,
	expected map[*autofunc.Variable]linalg.Vector) {
	if len(actual) != len(expected) {
		t.Errorf("%s: expected len %d got len %d", label, len(expected), len(actual))
		return
	}
	for variable, expectedVec := range expected {
		actualVec := actual[variable]
		if len(actualVec) != len(expectedVec) {
			t.Errorf("%s: variable len should be %d but got %d",
				label, len(expectedVec), len(actualVec))
			continue
		}
		for i, x := range expectedVec {
			a := actualVec[i]
			if math.Abs(a-x) > rnnSeqFuncTestPrec {
				t.Errorf("%s: entry %d: should be %f but got %f", label, i, x, a)
			}
		}
	}
}

func seqsToVarSeqs(s [][]linalg.Vector) [][]autofunc.Result {
	res := make([][]autofunc.Result, len(s))
	for i, x := range s {
		for _, v := range x {
			res[i] = append(res[i], &autofunc.Variable{Vector: v})
		}
	}
	return res
}

func seqsToRVarSeqs(s [][]linalg.Vector) [][]autofunc.RResult {
	rand.Seed(123)
	res := make([][]autofunc.RResult, len(s))
	for i, x := range s {
		for _, v := range x {
			variable := &autofunc.Variable{Vector: v}
			rVec := make(linalg.Vector, len(v))
			for i := range rVec {
				rVec[i] = rand.NormFloat64()
			}
			rVar := &autofunc.RVariable{Variable: variable, ROutputVec: rVec}
			res[i] = append(res[i], rVar)
		}
	}
	return res
}

func randomUpstreamSeqs(s [][]linalg.Vector, outputSize int) [][]linalg.Vector {
	rand.Seed(123123)
	var res [][]linalg.Vector
	for _, v := range s {
		randSeq := make([]linalg.Vector, len(v))
		for i := range v {
			randSeq[i] = make(linalg.Vector, outputSize)
			for j := range randSeq[i] {
				randSeq[i][j] = rand.NormFloat64()
			}
		}
		res = append(res, randSeq)
	}
	return res
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
