package rnntest

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

func TestLSTM(t *testing.T) {
	testVars := []*autofunc.Variable{
		{Vector: []float64{0.098591, -0.595453, -0.751214, 0.266051}},
		{Vector: []float64{0.988517, 0.107284, -0.331529, 0.028565}},
		{Vector: []float64{-0.150604, 0.889039, 0.120916, 0.240999}},
		{Vector: []float64{0.961058, 0.878608, 0.052284, -0.635746}},
	}
	testSeqs := [][]*autofunc.Variable{
		{testVars[0], testVars[2]},
		{testVars[1]},
		{testVars[2], testVars[1], testVars[3]},
	}
	testRV := autofunc.RVector{
		testVars[0]: []float64{0.62524, -0.52979, 0.33020, 0.54462},
		testVars[1]: []float64{0.13498, 0.12607, 0.35989, 0.23255},
		testVars[2]: []float64{0.85996, 0.68435, -0.68506, 0.96907},
		testVars[3]: []float64{-0.79095, -0.33867, 0.86759, -0.16159},
	}
	block := rnn.NewLSTM(4, 2)
	for _, v := range block.Parameters() {
		testVars = append(testVars, v)
		testRV[v] = make(linalg.Vector, len(v.Vector))
		for i := range v.Vector {
			testRV[v][i] = rand.NormFloat64()
		}
	}
	checker := &BlockChecker{
		B:     block,
		Input: testSeqs,
		Vars:  testVars,
		RV:    testRV,
	}
	checker.FullCheck(t)
}
