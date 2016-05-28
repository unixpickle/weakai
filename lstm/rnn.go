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

	inputs      []linalg.Vector
	lstmOutputs []*LSTMOutput
	outputs     []linalg.Vector

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

	output := r.memoryParams.PropagateForward(r.currentState, in)
	r.currentState = output.NewState

	augmentedInput := &linalg.Matrix{
		Rows: len(in) + len(output.MaskedState),
		Cols: 1,
		Data: make([]float64, len(in)+len(output.MaskedState)),
	}
	copy(augmentedInput.Data, in)
	copy(augmentedInput.Data[len(in):], output.MaskedState)
	result := linalg.Vector(r.outWeights.Mul(augmentedInput).Data).Add(r.outBiases)
	sigmoidAll(result)

	r.outputs = append(r.outputs, result)
	r.lstmOutputs = append(r.lstmOutputs, output)

	return result
}

func (r *RNN) CostGradient(cost CostFunc) *Gradient {
	grad := NewGradient(r.memoryParams.InputSize, r.memoryParams.StateSize, len(r.outBiases))

	costPartials := make([]linalg.Vector, len(r.outputs))
	for i, out := range r.outputs {
		costPartials[i] = make(linalg.Vector, len(out))
		cost.Gradient(out, r.expectedOutputs[i], costPartials[i])
	}

	r.computeOutputPartials(grad, costPartials)

	return grad
}

func (r *RNN) computeOutputPartials(g *Gradient, costPartials []linalg.Vector) {
	inputCount := r.memoryParams.InputSize
	hiddenCount := r.memoryParams.StateSize

	for t, partial := range costPartials {
		for neuronIdx, costPartial := range partial {
			neuronOut := r.outputs[t][neuronIdx]
			sigmoidPartial := (neuronOut - 1) * neuronOut
			sumPartial := sigmoidPartial * costPartial

			g.OutBiases[neuronIdx] += sumPartial
			for inputIdx := 0; inputIdx < inputCount; inputIdx++ {
				val := g.OutWeights.Get(neuronIdx, inputIdx)
				val += r.inputs[t][inputIdx] * sumPartial
				g.OutWeights.Set(neuronIdx, inputIdx, val)
			}
			for hiddenIdx := 0; hiddenIdx < hiddenCount; hiddenIdx++ {
				col := hiddenIdx + inputCount
				val := g.OutWeights.Get(neuronIdx, col)
				val += r.lstmOutputs[t].MaskedState[hiddenIdx] * sumPartial
				g.OutWeights.Set(neuronIdx, col, val)

				weightVal := r.outWeights.Get(neuronIdx, col)
				stateVal := r.lstmOutputs[t].NewState[hiddenIdx]
				maskVal := r.lstmOutputs[t].OutputMask[hiddenIdx]
				maskSigmoidPartial := (maskVal - 1) * maskVal
				maskSumPartial := maskSigmoidPartial * weightVal * stateVal * sumPartial
				g.OutGateBiases[hiddenIdx] += maskSumPartial
				for inputIdx1 := 0; inputIdx1 < inputCount; inputIdx1++ {
					val := g.OutGate.Get(neuronIdx, inputIdx1)
					val += r.inputs[t][inputIdx1] * maskSumPartial
					g.OutGate.Set(neuronIdx, inputIdx1, val)
				}
				if t > 0 {
					for hiddenIdx1 := 0; hiddenIdx1 < hiddenCount; hiddenIdx1++ {
						col1 := hiddenIdx1 + inputCount
						val := g.OutGate.Get(neuronIdx, col1)
						val += r.lstmOutputs[t-1].NewState[hiddenIdx1] * maskSumPartial
						g.OutGate.Set(neuronIdx, col1, val)
					}
				}
			}
		}
	}
}
