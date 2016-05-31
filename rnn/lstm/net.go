package lstm

import (
	"math"
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

// An Net is a single-layer recurrent neural net
// with LSTM hidden units.
type Net struct {
	activation rnn.ActivationFunc

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
}

func NewNet(activation rnn.ActivationFunc, inputSize, stateSize, outputSize int) *Net {
	return &Net{
		activation:   activation,
		outWeights:   linalg.NewMatrix(outputSize, inputSize+stateSize),
		outBiases:    make(linalg.Vector, outputSize),
		memoryParams: NewLSTM(inputSize, stateSize),
		currentState: make(linalg.Vector, stateSize),
	}
}

// Randomize randomly initializes the output
// and LSTM parameters.
func (r *Net) Randomize() {
	weightCoeff := math.Sqrt(3.0 / float64(r.outWeights.Cols))
	for i := range r.outWeights.Data {
		r.outWeights.Data[i] = (rand.Float64()*2 - 1) * weightCoeff
	}
	r.memoryParams.Randomize()
}

// StepTime gives the Net another input and
// returns the Net's output for that input.
func (r *Net) StepTime(in linalg.Vector) linalg.Vector {
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
	for i, x := range result {
		result[i] = r.activation.Eval(x)
	}

	r.outputs = append(r.outputs, result)
	r.lstmOutputs = append(r.lstmOutputs, output)

	return result
}

// CostGradient returns the gradient of the cost
// with respect to all of the Net parameters.
// The costPartials argument specifies the partial
// derivatives of the cost function with respect to
// each of the outputs for each of the time steps
// performed on this network.
func (r *Net) CostGradient(costPartials []linalg.Vector) rnn.Gradient {
	grad := NewGradient(r.memoryParams.InputSize, r.memoryParams.StateSize, len(r.outBiases),
		len(r.inputs))

	r.computeShallowPartials(grad, costPartials)
	r.computeDeepPartials(grad, costPartials)

	return grad
}

// StepGradient updates the parameters of the Net
// by adding values from the given gradient.
//
// This automatically resets the Net as if Reset
// were called on it, since the old forward-propagated
// outputs will be inaccurate after stepping.
//
// To perform gradient descent, you should negate
// the gradient and scale it down using its Scale()
// method before calling StepGradient().
func (r *Net) StepGradient(gInterface rnn.Gradient) {
	g := gInterface.(*Gradient)
	r.outWeights.Add(g.OutWeights)
	r.outBiases.Add(g.OutBiases)
	r.memoryParams.InWeights.Add(g.InWeights)
	r.memoryParams.InGate.Add(g.InGate)
	r.memoryParams.RemGate.Add(g.RemGate)
	r.memoryParams.OutGate.Add(g.OutGate)
	r.memoryParams.InBiases.Add(g.InBiases)
	r.memoryParams.InGateBiases.Add(g.InGateBiases)
	r.memoryParams.RemGateBiases.Add(g.RemGateBiases)
	r.memoryParams.OutGateBiases.Add(g.OutGateBiases)
	r.Reset()
}

// Reset goes back to time 0 and erases all the
// previous inputs and states.
// This does not reset weights or biases.
func (r *Net) Reset() {
	r.inputs = nil
	r.lstmOutputs = nil
	r.outputs = nil
	r.currentState = make(linalg.Vector, r.memoryParams.StateSize)
}

func (r *Net) computeShallowPartials(g *Gradient, costPartials []linalg.Vector) {
	inputCount := r.memoryParams.InputSize
	hiddenCount := r.memoryParams.StateSize

	for t, partial := range costPartials {
		for neuronIdx, costPartial := range partial {
			neuronOut := r.outputs[t][neuronIdx]
			activationPartial := r.activation.Deriv(neuronOut)
			sumPartial := activationPartial * costPartial

			g.OutBiases[neuronIdx] += sumPartial
			for inputIdx := 0; inputIdx < inputCount; inputIdx++ {
				val := g.OutWeights.Get(neuronIdx, inputIdx)
				val += r.inputs[t][inputIdx] * sumPartial
				g.OutWeights.Set(neuronIdx, inputIdx, val)

				weight := r.outWeights.Get(neuronIdx, inputIdx)
				g.InputGrads[t][inputIdx] += weight * sumPartial
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

func (r *Net) computeDeepPartials(g *Gradient, costPartials []linalg.Vector) {
	hiddenCount := r.memoryParams.StateSize
	upstreamStateGrad := make(linalg.Vector, hiddenCount)

	for t := len(costPartials) - 1; t >= 0; t-- {
		statePartial, olderPartial, inGrad := r.localStateGrads(costPartials[t], t)
		statePartial.Add(upstreamStateGrad)
		upstreamStateGrad = olderPartial
		g.InputGrads[t].Add(inGrad)

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

// localStateGrads computes three gradients:
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
//
// 3) similar to 2), except with respect to the input
// vector at time t instead of the previous state.
func (r *Net) localStateGrads(costGrad linalg.Vector, t int) (current, older,
	inGrad linalg.Vector) {
	hiddenCount := r.memoryParams.StateSize
	inputCount := r.memoryParams.InputSize
	current = make(linalg.Vector, hiddenCount)
	inGrad = make(linalg.Vector, inputCount)

	if t > 0 {
		older = make(linalg.Vector, hiddenCount)
	}

	for neuronIdx, partial := range costGrad {
		output := r.outputs[t][neuronIdx]
		activationDeriv := r.activation.Deriv(output)
		sumDeriv := activationDeriv * partial
		for hiddenIdx := range current {
			col := hiddenIdx + inputCount
			weight := r.outWeights.Get(neuronIdx, col)
			outMask := r.lstmOutputs[t].OutputMask[hiddenIdx]
			current[hiddenIdx] += outMask * weight * sumDeriv

			stateVal := r.lstmOutputs[t].NewState[hiddenIdx]
			maskSigmoidPartial := outMask * (1 - outMask)
			maskSumPartial := maskSigmoidPartial * stateVal * weight * sumDeriv
			for inputIdx := range inGrad {
				val := r.memoryParams.OutGate.Get(hiddenIdx, inputIdx)
				inGrad[inputIdx] += val * maskSumPartial
			}

			if t == 0 {
				continue
			}

			for hiddenIdx1 := range older {
				col := inputCount + hiddenIdx1
				val := r.memoryParams.OutGate.Get(hiddenIdx, col)
				older[hiddenIdx1] += val * maskSumPartial
			}
		}
	}

	return
}

func (r *Net) rememberGateGrad(g *Gradient, upstreamGrad, statePartial linalg.Vector, t int) {
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

			weight := r.memoryParams.RemGate.Get(hiddenIdx, inputIdx)
			g.InputGrads[t][inputIdx] += weight * maskSumPartial
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

func (r *Net) inputGateGrad(g *Gradient, upstreamGrad, statePartial linalg.Vector, t int) {
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

			weight := r.memoryParams.InGate.Get(hiddenIdx, inputIdx)
			g.InputGrads[t][inputIdx] += weight * maskSumPartial
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

func (r *Net) inputGrad(g *Gradient, upstreamGrad, statePartial linalg.Vector, t int) {
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

			weight := r.memoryParams.InWeights.Get(hiddenIdx, inputIdx)
			g.InputGrads[t][inputIdx] += weight * inputSumPartial
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
