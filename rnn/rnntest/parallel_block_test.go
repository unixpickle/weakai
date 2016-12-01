package rnntest

import (
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/weakai/rnn"
)

func TestParallelBlockOutput(t *testing.T) {
	block1 := rnn.NewLSTM(3, 3)
	block2 := rnn.NewLSTM(3, 4)
	input := []autofunc.Result{
		&autofunc.Variable{Vector: []float64{1, -1, 0.5}},
		&autofunc.Variable{Vector: []float64{-0.5, 1, 0.5}},
	}
	o1 := block1.ApplyBlock([]rnn.State{block1.StartState(), block1.StartState()}, input)
	o2 := block2.ApplyBlock([]rnn.State{block2.StartState(), block2.StartState()}, input)
	expected1 := append(o1.Outputs()[0], o2.Outputs()[0]...)
	expected2 := append(o1.Outputs()[1], o2.Outputs()[1]...)

	pb := rnn.ParallelBlock{block1, block2}
	pbOut := pb.ApplyBlock([]rnn.State{pb.StartState(), pb.StartState()}, input)
	actual1 := pbOut.Outputs()[0]
	actual2 := pbOut.Outputs()[1]

	if actual1.Copy().Scale(-1).Add(expected1).MaxAbs() > 1e-5 {
		t.Errorf("expected output1 %v but got %v", expected1, actual1)
	}
	if actual2.Copy().Scale(-1).Add(expected2).MaxAbs() > 1e-5 {
		t.Errorf("expected output2 %v but got %v", expected2, actual2)
	}
}

func TestParallelBlock(t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}
	block1 := rnn.NewLSTM(4, 3)
	block2 := rnn.NewLSTM(4, 1)
	block3 := rnn.NewLSTM(4, 2)
	pb := rnn.ParallelBlock{block1, block2, block3}
	ch := NewChecker4In(pb, pb)
	ch.FullCheck(t)
}
