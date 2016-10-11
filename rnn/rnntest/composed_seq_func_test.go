package rnntest

import (
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

func TestComposedSeqFuncOutput(t *testing.T) {
	seqFunc1 := &rnn.BlockSeqFunc{Block: rnn.NewLSTM(3, 2)}
	seqFunc2 := &rnn.BlockSeqFunc{Block: rnn.NewLSTM(2, 3)}
	inSeqs := [][]autofunc.Result{
		{
			&autofunc.Variable{Vector: []float64{1, 2, 3}},
			&autofunc.Variable{Vector: []float64{2, 1, 0}},
		},
		{
			&autofunc.Variable{Vector: []float64{-1, 0, 1}},
			&autofunc.Variable{Vector: []float64{1, -1, 0}},
			&autofunc.Variable{Vector: []float64{2, 1, -1}},
		},
	}
	composedFunc := autofunc.ComposedFunc{
		&rnn.SeqFuncFunc{S: seqFunc1, InSize: 3},
		&rnn.SeqFuncFunc{S: seqFunc2, InSize: 2},
	}
	composedSeqFunc := rnn.ComposedSeqFunc{seqFunc1, seqFunc2}

	for i, seq := range inSeqs {
		var rawSeq linalg.Vector
		for _, item := range seq {
			rawSeq = append(rawSeq, item.Output()...)
		}
		expected := composedFunc.Apply(&autofunc.Variable{Vector: rawSeq}).Output()
		actual := composedSeqFunc.BatchSeqs([][]autofunc.Result{seq})
		if len(actual.OutputSeqs()) != 1 {
			t.Errorf("seq %d: bad count %d", i, len(actual.OutputSeqs()))
			continue
		}
		outSeq := actual.OutputSeqs()[0]
		var joined linalg.Vector
		for _, x := range outSeq {
			joined = append(joined, x...)
		}
		if len(joined) != len(expected) {
			t.Errorf("seq %d: len should be %d but got %d", i, len(joined), len(expected))
			continue
		}
		diff := joined.Copy().Scale(-1).Add(expected).MaxAbs()
		if diff > 1e-5 {
			t.Errorf("seq %d: expected %v but got %v", i, expected, joined)
		}
	}
}

func TestComposedSeqFuncBatchOutput(t *testing.T) {
	inSeqs := [][]autofunc.Result{
		{
			&autofunc.Variable{Vector: []float64{1, 2, 3}},
			&autofunc.Variable{Vector: []float64{2, 1, 0}},
		},
		{
			&autofunc.Variable{Vector: []float64{-1, 0, 1}},
			&autofunc.Variable{Vector: []float64{1, -1, 0}},
			&autofunc.Variable{Vector: []float64{2, 1, -1}},
		},
		{
			&autofunc.Variable{Vector: []float64{0, 1, -1.5}},
		},
	}
	seqFunc1 := &rnn.BlockSeqFunc{Block: rnn.NewLSTM(3, 2)}
	seqFunc2 := &rnn.BlockSeqFunc{Block: rnn.NewLSTM(2, 3)}
	composedSeqFunc := rnn.ComposedSeqFunc{seqFunc1, seqFunc2}

	actual := composedSeqFunc.BatchSeqs(inSeqs).OutputSeqs()
	var expected [][]linalg.Vector
	for _, seq := range inSeqs {
		out := composedSeqFunc.BatchSeqs([][]autofunc.Result{seq})
		expected = append(expected, out.OutputSeqs()...)
	}

	if len(actual) != len(expected) {
		t.Fatalf("expected %d sequences but got %d", len(expected), len(actual))
	}

	for i, act := range actual {
		exp := expected[i]
		if len(act) != len(exp) {
			t.Errorf("seq %d: len should be %d but it's %d", i, len(exp), len(act))
			continue
		}
		for j, a := range act {
			x := exp[j]
			diff := a.Copy().Scale(-1).Add(x)
			if diff.MaxAbs() > 1e-5 {
				t.Errorf("seq %d entry %d: should be %v but it's %v", i, j, x, a)
			}
		}
	}
}

func TestComposedSeqFuncGradients(t *testing.T) {
	seqFunc1 := &rnn.BlockSeqFunc{Block: rnn.NewLSTM(3, 2)}
	seqFunc2 := &rnn.BlockSeqFunc{Block: rnn.NewLSTM(2, 3)}
	composedSeqFunc := rnn.ComposedSeqFunc{seqFunc1, seqFunc2}
	params := composedSeqFunc.Parameters()

	tester := SeqFuncTest{
		S:      composedSeqFunc,
		Params: params,
		TestSeqs: [][]linalg.Vector{
			{{1, 2, 3}, {2, 1, 0}, {-1, 0, 1}},
			{{3, -3, 3}, {0.5, 0.3, 0.2}},
		},
	}
	tester.Run(t)
}
