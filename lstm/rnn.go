package lstm

import "github.com/unixpickle/num-analysis/linalg"

type RNN struct {
	// Columns of this matrix correspond first to
	// input values and then to values of the
	// masked output state.
	outWeights *linalg.Matrix

	outBiases linalg.Vector

	stateSize int

	// prototype is a template for all LSTM blocks
	// in this network.
	// It is aliased again and again for each time
	// iteration of the network.
	prototype *LSTM

	timeLayers      []*LSTM
	outputs         []linalg.Vector
	expectedOutputs []linalg.Vector
}

func NewRNN(inputSize, stateSize, outputSize int) *RNN {
	return &RNN{
		outWeights: linalg.NewMatrix(outputSize, inputSize+stateSize),
		outBiases:  make(linalg.Vector, outputSize),
		stateSize:  stateSize,
		prototype:  NewLSTM(inputSize, stateSize),
	}
}

func (r *RNN) StepTime(in, out linalg.Vector) linalg.Vector {
	r.expectedOutputs = append(r.expectedOutputs, out)

	layer := r.prototype.Alias()
	if len(r.timeLayers) > 0 {
		lastLayer := r.timeLayers[len(r.timeLayers)-1]
		layer.StateIn = lastLayer.NewState()
	} else {
		layer.StateIn = make(linalg.Vector, r.stateSize)
	}
	layer.Input = in

	layer.PropagateForward()
	stateOut := layer.StateOut()

	augmentedInput := &linalg.Matrix{
		Rows: len(in) + len(stateOut),
		Cols: 1,
		Data: make([]float64, len(in)+len(stateOut)),
	}
	copy(augmentedInput.Data, in)
	copy(augmentedInput.Data[len(in):], stateOut)
	result := linalg.Vector(r.outWeights.Mul(augmentedInput).Data).Add(r.outBiases)

	r.timeLayers = append(r.timeLayers, layer)
	r.outputs = append(r.outputs, result)

	return result
}
