package rnntest

import (
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

func TestStateOutBlock(t *testing.T) {
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
	net := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  8,
			OutputCount: 4,
		},
		&neuralnet.HyperbolicTangent{},
	}
	net.Randomize()
	block := &rnn.StateOutBlock{
		Block: &rnn.BatcherBlock{B: net.BatchLearner(), StateSize: 4, Start: testVars[2]},
	}
	checker := &BlockChecker{
		B:     block,
		Input: testSeqs,
		Vars:  testVars,
		RV:    testRV,
	}
	checker.FullCheck(t)
}
