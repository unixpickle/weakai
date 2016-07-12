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
		expected = append(expected, evaluateROutputs(rVec, block, inSeq))
	}

	testSequencesEqual(t, actual, expected)
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

func evaluateROutputs(rv autofunc.RVector, b rnn.Block,
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
