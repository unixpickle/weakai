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

const npPowerIterations = 1000

// NPRNN is an RNN that implements the architecture
// described in http://arxiv.org/pdf/1511.03771v3.pdf.
// An NPRNN outputs its current state, so that Blocks
// may be stacked on top of it to use this state.
type NPRNN interface {
	Block
	serializer.Serializer
	sgd.Learner
}

// NewNPRNN creates an np-RNN with the given input
// and hidden sizes.
func NewNPRNN(inputSize, hiddenSize int) NPRNN {
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

	normalMat := linalg.NewMatrix(hiddenSize, hiddenSize)
	for i := range normalMat.Data {
		normalMat.Data[i] = rand.NormFloat64()
	}
	normalMat = normalMat.Mul(normalMat.Transpose())
	normalMat.Scale(1 / float64(hiddenSize))
	for i := 0; i < hiddenSize; i++ {
		normalMat.Set(i, i, 1+normalMat.Get(i, i))
	}
	normalMat.Scale(1 / maxEigenvalue(normalMat))

	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < hiddenSize; j++ {
			x := normalMat.Get(i, j)
			weights[j+inputSize+i*(hiddenSize+inputSize)] = x
		}
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

func maxEigenvalue(m *linalg.Matrix) float64 {
	inVec := make(linalg.Vector, m.Rows)
	for i := range inVec {
		inVec[i] = rand.NormFloat64()
	}
	inMat := linalg.NewMatrixColumn(inVec)
	for i := 0; i < npPowerIterations; i++ {
		inMat = m.MulFast(inMat)
		vec := linalg.Vector(inMat.Data)
		vec.Scale(1 / vec.Mag())
	}
	outVec := linalg.Vector(m.MulFast(inMat).Data)
	return outVec.Mag()
}
