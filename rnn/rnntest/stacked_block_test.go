package rnntest

import (
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

func TestStackedBlock(t *testing.T) {
	testVars := []*autofunc.Variable{
		{Vector: []float64{0.098591, -0.595453, -0.751214, 0.266051}},
		{Vector: []float64{0.988517, 0.107284, -0.331529, 0.028565}},
		{Vector: []float64{-0.150604, 0.889039, 0.120916, 0.240999}},
		{Vector: []float64{0.961058, 0.878608, 0.052284, -0.635746}},
		{Vector: []float64{0.31415, -0.2718}},
		{Vector: []float64{-0.6}},
	}
	testSeqs := [][]*autofunc.Variable{
		{testVars[0], testVars[2]},
		{testVars[1]},
		{testVars[2], testVars[1], testVars[3]},
	}
	testRV := autofunc.RVector{
		testVars[0]: []float64{0.62524, 0.52979, 0.33020, 0.54462},
		testVars[1]: []float64{0.13498, 0.12607, 0.35989, 0.23255},
		testVars[2]: []float64{0.85996, 0.68435, 0.68506, 0.96907},
		testVars[3]: []float64{0.79095, 0.33867, 0.86759, 0.16159},
		testVars[4]: []float64{-0.79095, 0.33867},
		testVars[5]: []float64{0.33867},
	}
	net1 := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  6,
			OutputCount: 6,
		},
		&neuralnet.HyperbolicTangent{},
	}
	net1.Randomize()
	net2 := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  5,
			OutputCount: 5,
		},
		&neuralnet.HyperbolicTangent{},
	}
	net2.Randomize()
	block := &rnn.StackedBlock{
		&rnn.BatcherBlock{B: net1.BatchLearner(), StateSize: 2, Start: testVars[4]},
		&rnn.BatcherBlock{B: net2.BatchLearner(), StateSize: 1, Start: testVars[5]},
	}
	checker := &BlockChecker{
		B:     block,
		Input: testSeqs,
		Vars:  testVars,
		RV:    testRV,
	}
	checker.FullCheck(t)
}
