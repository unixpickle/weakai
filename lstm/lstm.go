package lstm

import (
	"math"

	"github.com/unixpickle/num-analysis/linalg"
)

// LSTM is one instance of an LSTM layer.
type LSTM struct {
	// These are input vectors which should be set
	// externally.
	StateIn linalg.Vector
	Input   linalg.Vector

	// The columns in these matrices correspond
	// first to inputs, then to previous states.
	// The rows correspond to different neurons
	// in the hidden (state) layer.
	inWeights *linalg.Matrix
	inGate    *linalg.Matrix
	remGate   *linalg.Matrix
	outGate   *linalg.Matrix

	// Each element of these vectors corresponds
	// to a hidden neuron's index.
	inBiases      linalg.Vector
	inGateBiases  linalg.Vector
	remGateBiases linalg.Vector
	outGateBiases linalg.Vector

	// newState is an output vector which is set
	// by NewLSTM to store the internal state
	// resulting from forward propagation.
	newState linalg.Vector

	// stateOut is an output vector which is set
	// by NewLSTM to store the gated output of
	// the layer.
	stateOut linalg.Vector
}

func NewLSTM(inputSize, stateSize int) *LSTM {
	return &LSTM{
		inWeights: linalg.NewMatrix(stateSize, inputSize+stateSize),
		inGate:    linalg.NewMatrix(stateSize, inputSize+stateSize),
		remGate:   linalg.NewMatrix(stateSize, inputSize+stateSize),
		outGate:   linalg.NewMatrix(stateSize, inputSize+stateSize),

		inBiases:      make(linalg.Vector, stateSize),
		inGateBiases:  make(linalg.Vector, stateSize),
		remGateBiases: make(linalg.Vector, stateSize),
		outGateBiases: make(linalg.Vector, stateSize),

		newState: make(linalg.Vector, stateSize),
		stateOut: make(linalg.Vector, stateSize),
	}
}

func (l *LSTM) PropagateForward() {
	augInSize := len(l.StateIn) + len(l.Input)
	inputMat := &linalg.Matrix{
		Rows: augInSize,
		Cols: 1,
		Data: make([]float64, augInSize),
	}
	copy(inputMat.Data, l.Input)
	copy(inputMat.Data[len(l.Input):], l.StateIn)

	inputVal := linalg.Vector(l.inWeights.Mul(inputMat).Data).Add(l.inBiases)
	inputMask := linalg.Vector(l.inGate.Mul(inputMat).Data).Add(l.inGateBiases)
	memoryMask := linalg.Vector(l.remGate.Mul(inputMat).Data).Add(l.remGateBiases)
	outputMask := linalg.Vector(l.outGate.Mul(inputMat).Data).Add(l.outGateBiases)

	sigmoidAll(inputVal, inputMask, memoryMask, outputMask)

	piecewiseMul(inputVal, inputMask)
	for i, m := range memoryMask {
		l.newState[i] = m * l.StateIn[i]
	}

	piecewiseMul(l.newState, memoryMask)
	l.newState.Add(inputVal)
	copy(l.stateOut, l.newState)
	piecewiseMul(l.stateOut, outputMask)
}

func (l *LSTM) Alias() *LSTM {
	res := *l
	res.newState = make(linalg.Vector, len(l.newState))
	res.stateOut = make(linalg.Vector, len(l.stateOut))
	res.StateIn = nil
	res.Input = nil
	return &res
}

// NewState returns a vector which forward propagation
// fills with the new internal state of the LSTM unit.
func (l *LSTM) NewState() linalg.Vector {
	return l.newState
}

// StateOut returns a vector which forward propagation
// fills with the new, masked output state of the unit.
func (l *LSTM) StateOut() linalg.Vector {
	return l.stateOut
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
