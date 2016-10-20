package rnntest

import (
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

func TestStateOutBlock(t *testing.T) {
	net := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  8,
			OutputCount: 4,
		},
		&neuralnet.HyperbolicTangent{},
	}
	net.Randomize()
	startVar := &autofunc.Variable{Vector: []float64{0.3, -0.3, 0.2, 0.5}}
	block := &rnn.StateOutBlock{
		Block: &rnn.BatcherBlock{
			B:         net.BatchLearner(),
			StateSize: 4,
			Start:     startVar,
		},
	}
	learner := append(stateOutBlockLearner{startVar}, net.Parameters()...)
	NewChecker4In(block, learner).FullCheck(t)
}

type stateOutBlockLearner []*autofunc.Variable

func (s stateOutBlockLearner) Parameters() []*autofunc.Variable {
	return s
}
