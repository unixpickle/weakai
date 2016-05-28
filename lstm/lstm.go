package lstm

import (
	"math"

	"github.com/unixpickle/num-analysis/linalg"
)

// LSTM stores the parameters of a
// Long Short-Term Memory layer.
type LSTM struct {
	// The columns in these matrices correspond
	// first to inputs, then to the last state.
	// The rows correspond to different neurons
	// in the hidden (state) layer.
	InWeights *linalg.Matrix
	InGate    *linalg.Matrix
	RemGate   *linalg.Matrix
	OutGate   *linalg.Matrix

	// Each element of these vectors corresponds
	// to one hidden gated neuron.
	InBiases      linalg.Vector
	InGateBiases  linalg.Vector
	RemGateBiases linalg.Vector
	OutGateBiases linalg.Vector

	InputSize int
	StateSize int
}

func NewLSTM(inputSize, stateSize int) *LSTM {
	return &LSTM{
		InWeights: linalg.NewMatrix(stateSize, inputSize+stateSize),
		InGate:    linalg.NewMatrix(stateSize, inputSize+stateSize),
		RemGate:   linalg.NewMatrix(stateSize, inputSize+stateSize),
		OutGate:   linalg.NewMatrix(stateSize, inputSize+stateSize),

		InBiases:      make(linalg.Vector, stateSize),
		InGateBiases:  make(linalg.Vector, stateSize),
		RemGateBiases: make(linalg.Vector, stateSize),
		OutGateBiases: make(linalg.Vector, stateSize),

		InputSize: inputSize,
		StateSize: stateSize,
	}
}

// PropagateForward takes a state and an input and
// generates a new state and a masked, output-ready
// version of that state.
func (l *LSTM) PropagateForward(state, input linalg.Vector) (newState, masked linalg.Vector) {
	augInSize := len(state) + len(input)
	inputMat := &linalg.Matrix{
		Rows: augInSize,
		Cols: 1,
		Data: make([]float64, augInSize),
	}
	copy(inputMat.Data, input)
	copy(inputMat.Data[len(input):], state)

	inputVal := linalg.Vector(l.InWeights.Mul(inputMat).Data).Add(l.InBiases)
	inputMask := linalg.Vector(l.InGate.Mul(inputMat).Data).Add(l.InGateBiases)
	memoryMask := linalg.Vector(l.RemGate.Mul(inputMat).Data).Add(l.RemGateBiases)
	outputMask := linalg.Vector(l.OutGate.Mul(inputMat).Data).Add(l.OutGateBiases)

	sigmoidAll(inputVal, inputMask, memoryMask, outputMask)
	piecewiseMul(inputVal, inputMask)

	masked = make(linalg.Vector, len(state))
	newState = make(linalg.Vector, len(state))
	for i, m := range memoryMask {
		newState[i] = m * state[i]
	}

	piecewiseMul(newState, memoryMask)
	newState.Add(inputVal)
	copy(masked, newState)
	piecewiseMul(masked, outputMask)

	return
}

func piecewiseMul(target linalg.Vector, mask linalg.Vector) {
	for i, m := range mask {
		target[i] *= m
	}
}

func sigmoidAll(vs ...linalg.Vector) {
	for _, v := range vs {
		for i, x := range v {
			v[i] = sigmoid(x)
		}
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(x))
}
