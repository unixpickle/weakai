package lstm

import "github.com/unixpickle/num-analysis/linalg"

type RNN struct {
	// The columns in this matrix correspond first
	// to input values and then to state values.
	outWeights *linalg.Matrix

	// The elements in this vector correspond to
	// elements of the output.
	outBiases linalg.Vector

	memoryParams *LSTM
	currentState linalg.Vector

	inputs  []linalg.Vector
	outputs []linalg.Vector

	expectedOutputs []linalg.Vector
}

func NewRNN(inputSize, stateSize, outputSize int) *RNN {
	return &RNN{
		outWeights:   linalg.NewMatrix(outputSize, inputSize+stateSize),
		outBiases:    make(linalg.Vector, outputSize),
		memoryParams: NewLSTM(inputSize, stateSize),
		currentState: make(linalg.Vector, stateSize),
	}
}

func (r *RNN) StepTime(in, out linalg.Vector) linalg.Vector {
	r.expectedOutputs = append(r.expectedOutputs, out)
	r.inputs = append(r.inputs, in)
	r.expectedOutputs = append(r.expectedOutputs, out)

	newState, masked := r.memoryParams.PropagateForward(r.currentState, in)
	r.currentState = newState

	augmentedInput := &linalg.Matrix{
		Rows: len(in) + len(masked),
		Cols: 1,
		Data: make([]float64, len(in)+len(masked)),
	}
	copy(augmentedInput.Data, in)
	copy(augmentedInput.Data[len(in):], masked)
	result := linalg.Vector(r.outWeights.Mul(augmentedInput).Data).Add(r.outBiases)

	r.outputs = append(r.outputs, result)

	return result
}
