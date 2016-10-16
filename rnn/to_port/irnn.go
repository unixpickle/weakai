package rnn

import (
	"math"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// IRNN is an RNN that implements the identity RNN
// described in https://arxiv.org/pdf/1504.00941v2.pdf.
// An IRNN outputs its current state, so that Blocks
// may be stacked on top of it to use this state.
type IRNN interface {
	Block
	serializer.Serializer
	sgd.Learner
}

// NewIRNN creates a block for an identity RNN with
// the given number of hidden units.
// The initial state connections are set to an identity
// matrix scaled by identityScale.
func NewIRNN(inputSize, hiddenSize int, identityScale float64) IRNN {
	matrix := &neuralnet.DenseLayer{
		InputCount:  inputSize + hiddenSize,
		OutputCount: hiddenSize,
		Biases: &autofunc.LinAdd{
			Var: &autofunc.Variable{
				Vector: make(linalg.Vector, hiddenSize),
			},
		},
		Weights: &autofunc.LinTran{
			Rows: hiddenSize,
			Cols: inputSize + hiddenSize,
			Data: &autofunc.Variable{
				Vector: make(linalg.Vector, hiddenSize*(inputSize+hiddenSize)),
			},
		},
	}
	weights := matrix.Weights.Data.Vector
	for i := 0; i < hiddenSize; i++ {
		weights[i+inputSize+i*(inputSize+hiddenSize)] = identityScale
	}

	weightScale := 1 / math.Sqrt(float64(inputSize))
	for j := 0; j < hiddenSize; j++ {
		for i := 0; i < inputSize; i++ {
			weights[i+j*(inputSize+hiddenSize)] = (rand.Float64()*2 - 1) * weightScale
		}
	}

	network := neuralnet.Network{
		matrix,
		&neuralnet.ReLU{},
	}

	return &StateOutBlock{
		Block: NewNetworkBlock(network, hiddenSize),
	}
}
