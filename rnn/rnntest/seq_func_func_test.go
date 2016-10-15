package rnntest

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

func TestSeqFuncFunc(t *testing.T) {
	rand.Seed(123)
	block := rnn.NewLSTM(3, 2)
	seqFunc := &rnn.BlockSeqFunc{Block: block}
	seqFuncFunc := &rnn.SeqFuncFunc{S: seqFunc, InSize: 3}

	inSeq := &autofunc.Variable{
		Vector: []float64{0.35044, 0.66894, -0.42222, 0.37096, -0.83984,
			0.90732, -0.63026, -0.14004, 0.77829},
	}

	fc := &functest.FuncChecker{
		F:     seqFuncFunc,
		Vars:  append(block.Parameters(), inSeq),
		Input: inSeq,
	}
	fc.FullCheck(t)

	rv := autofunc.RVector{}
	for _, variable := range fc.Vars {
		rv[variable] = make(linalg.Vector, len(variable.Vector))
		for i := range rv[variable] {
			rv[variable][i] = rand.NormFloat64()
		}
	}

	rfc := &functest.RFuncChecker{
		F:     seqFuncFunc,
		Vars:  append(block.Parameters(), inSeq),
		Input: inSeq,
		RV:    rv,
	}
	rfc.FullCheck(t)
}
