package rbf

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/num-analysis/linalg/leastsquares"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// LeastSquares uses a least-squares approach to train the
// output layer of a network.
//
// The s argument is a set of neuralnet.VectorSamples.
//
// The bs argument specifies the batch size for applying
// the network.
//
// The least squares solution is returned and can be used
// as n.OutLayer.
// The value of n.OutLayer may be nil, but the other
// layers of n should be set.
func LeastSquares(n *Network, s sgd.SampleSet, bs int) *neuralnet.DenseLayer {
	comp := autofunc.ComposedBatcher{
		n.DistLayer,
		&autofunc.FuncBatcher{
			F: autofunc.ComposedFunc{n.ScaleLayer, n.ExpLayer},
		},
	}

	mat := linalg.NewMatrix(s.Len(), n.DistLayer.NumCenters())
	var outData linalg.Vector
	for i := 0; i < s.Len(); i += bs {
		b := bs
		if b+i > s.Len() {
			b = s.Len() - i
		}

		var input linalg.Vector
		for j := 0; j < b; j++ {
			sample := s.GetSample(j + i).(neuralnet.VectorSample)
			input = append(input, sample.Input...)
			outData = append(outData, sample.Output...)
		}

		res := comp.Batch(&autofunc.Variable{Vector: input}, b)
		copy(mat.Data[i*mat.Cols:], res.Output())
	}

	outMat := linalg.NewMatrix(s.Len(), len(outData)/s.Len())
	outMat.Data = outData

	solver := leastsquares.NewSolver(mat)
	var solutionData linalg.Vector
	for col := 0; col < outMat.Cols; col++ {
		outVec := outMat.Col(col)
		solution := solver.Solve(outVec)
		solutionData = append(solutionData, solution...)
	}

	resLayer := neuralnet.NewDenseLayer(n.DistLayer.NumCenters(), outMat.Cols)
	copy(resLayer.Weights.Data.Vector, solutionData)
	for i := range resLayer.Biases.Var.Vector {
		resLayer.Biases.Var.Vector[i] = 0
	}
	return resLayer
}
