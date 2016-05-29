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
	r.computeTimedPartials(grad, costPartials)

	return grad
}

func (r *RNN) computeOutputPartials(g *Gradient, costPartials []linalg.Vector) {
	inputCount := r.memoryParams.InputSize
	hiddenCount := r.memoryParams.StateSize

	for t, partial := range costPartials {
		for neuronIdx, costPartial := range partial {
			neuronOut := r.outputs[t][neuronIdx]
			sigmoidPartial := (1 - neuronOut) * neuronOut
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
				maskSigmoidPartial := (1 - maskVal) * maskVal
				maskSumPartial := maskSigmoidPartial * weightVal * stateVal * sumPartial
				g.OutGateBiases[hiddenIdx] += maskSumPartial
				for inputIdx1 := 0; inputIdx1 < inputCount; inputIdx1++ {
					val := g.OutGate.Get(hiddenIdx, inputIdx1)
					val += r.inputs[t][inputIdx1] * maskSumPartial
					g.OutGate.Set(hiddenIdx, inputIdx1, val)
				}
				if t > 0 {
					for hiddenIdx1 := 0; hiddenIdx1 < hiddenCount; hiddenIdx1++ {
						col1 := hiddenIdx1 + inputCount
						val := g.OutGate.Get(hiddenIdx, col1)
						val += r.lstmOutputs[t-1].NewState[hiddenIdx1] * maskSumPartial
						g.OutGate.Set(hiddenIdx, col1, val)
					}
				}
			}
		}
	}
}

func (r *RNN) computeTimedPartials(g *Gradient, costPartials []linalg.Vector) {
	hiddenCount := r.memoryParams.StateSize
	upstreamStateGrad := make(linalg.Vector, hiddenCount)

	for t := len(costPartials) - 1; t >= 0; t-- {
		statePartial, olderPartial := r.localStateGrads(costPartials[t], t)
		statePartial.Add(upstreamStateGrad)
		upstreamStateGrad = olderPartial

		if t > 0 {
			for hiddenIdx, rememberMask := range r.lstmOutputs[t].RememberMask {
				upstreamStateGrad[hiddenIdx] += rememberMask * statePartial[hiddenIdx]
			}
			r.rememberGateGrad(g, upstreamStateGrad, statePartial, t)
		}

		r.inputGateGrad(g, upstreamStateGrad, statePartial, t)
		r.inputGrad(g, upstreamStateGrad, statePartial, t)
	}
}

// localStateGrads computes two gradients:
//
// 1) the gradient of the part of the cost function
// influenced directly from the output at time t
// with respect to the state output at time t
// (before masking).
//
// 2) part of the gradient of the part of the cost
// function influenced directly by the output at
// time t with respect to the state from time t-1.
// This is only "part" of the gradient because it only
// accounts for the contribution of state t-1 to the
// output mask and thus to the output at time t.
func (r *RNN) localStateGrads(costGrad linalg.Vector, t int) (current, older linalg.Vector) {
	hiddenCount := r.memoryParams.StateSize
	inputCount := r.memoryParams.InputSize
	current = make(linalg.Vector, hiddenCount)

	if t > 0 {
		older = make(linalg.Vector, hiddenCount)
	}

	for neuronIdx, partial := range costGrad {
		output := r.outputs[t][neuronIdx]
		sigmoidDeriv := output * (1 - output)
		sumDeriv := sigmoidDeriv * partial
		for hiddenIdx := range current {
			col := hiddenIdx + inputCount
			weight := r.outWeights.Get(neuronIdx, col)
			outMask := r.lstmOutputs[t].OutputMask[hiddenIdx]
			current[hiddenIdx] += outMask * weight * sumDeriv

			if t == 0 {
				continue
			}

			stateVal := r.lstmOutputs[t].NewState[hiddenIdx]
			maskSigmoidPartial := outMask * (1 - outMask)
			maskSumPartial := maskSigmoidPartial * stateVal * weight * sumDeriv
			for hiddenIdx1 := range older {
				col := inputCount + hiddenIdx1
				val := r.memoryParams.OutGate.Get(hiddenIdx, col)
				older[hiddenIdx1] += val * maskSumPartial
			}
		}
	}

	return
}

func (r *RNN) rememberGateGrad(g *Gradient, upstreamGrad, statePartial linalg.Vector, t int) {
	for hiddenIdx, partial := range statePartial {
		lastValue := r.lstmOutputs[t-1].NewState[hiddenIdx]
		mask := r.lstmOutputs[t].RememberMask[hiddenIdx]
		maskSigmoidPartial := mask * (1 - mask)
		maskSumPartial := maskSigmoidPartial * lastValue * partial

		g.RemGateBiases[hiddenIdx] += maskSumPartial
		for inputIdx, inVal := range r.inputs[t] {
			val := g.RemGate.Get(hiddenIdx, inputIdx)
			val += inVal * maskSumPartial
			g.RemGate.Set(hiddenIdx, inputIdx, val)
		}

		for hiddenIdx1, inVal := range r.lstmOutputs[t-1].NewState {
			col := hiddenIdx1 + r.memoryParams.InputSize
			val := g.RemGate.Get(hiddenIdx, col)
			val += inVal * maskSumPartial
			g.RemGate.Set(hiddenIdx, col, val)

			weight := r.memoryParams.RemGate.Get(hiddenIdx, col)
			upstreamGrad[hiddenIdx1] += weight * maskSumPartial
		}
	}
}

func (r *RNN) inputGateGrad(g *Gradient, upstreamGrad, statePartial linalg.Vector, t int) {
	for hiddenIdx, partial := range statePartial {
		inputValue := r.lstmOutputs[t].MemInput[hiddenIdx]
		mask := r.lstmOutputs[t].InputMask[hiddenIdx]
		maskSigmoidPartial := mask * (1 - mask)
		maskSumPartial := maskSigmoidPartial * inputValue * partial

		g.InGateBiases[hiddenIdx] += maskSumPartial
		for inputIdx, inVal := range r.inputs[t] {
			val := g.InGate.Get(hiddenIdx, inputIdx)
			val += inVal * maskSumPartial
			g.InGate.Set(hiddenIdx, inputIdx, val)
		}

		if t == 0 {
			continue
		}

		for hiddenIdx1, inVal := range r.lstmOutputs[t-1].NewState {
			col := hiddenIdx1 + r.memoryParams.InputSize
			val := g.InGate.Get(hiddenIdx, col)
			val += inVal * maskSumPartial
			g.InGate.Set(hiddenIdx, col, val)

			weight := r.memoryParams.InGate.Get(hiddenIdx, col)
			upstreamGrad[hiddenIdx1] += weight * maskSumPartial
		}
	}
}

func (r *RNN) inputGrad(g *Gradient, upstreamGrad, statePartial linalg.Vector, t int) {
	for hiddenIdx, partial := range statePartial {
		mask := r.lstmOutputs[t].InputMask[hiddenIdx]
		inputValue := r.lstmOutputs[t].MemInput[hiddenIdx]
		inputSigmoidPartial := inputValue * (1 - inputValue)
		inputSumPartial := inputSigmoidPartial * mask * partial

		g.InBiases[hiddenIdx] += inputSumPartial
		for inputIdx, inVal := range r.inputs[t] {
			val := g.InWeights.Get(hiddenIdx, inputIdx)
			val += inVal * inputSumPartial
			g.InWeights.Set(hiddenIdx, inputIdx, val)
		}

		if t == 0 {
			continue
		}

		for hiddenIdx1, inVal := range r.lstmOutputs[t-1].NewState {
			col := hiddenIdx1 + r.memoryParams.InputSize
			val := g.InWeights.Get(hiddenIdx, col)
			val += inVal * inputSumPartial
			g.InWeights.Set(hiddenIdx, col, val)

			weight := r.memoryParams.InWeights.Get(hiddenIdx, col)
			upstreamGrad[hiddenIdx1] += weight * inputSumPartial
		}
	}
}
