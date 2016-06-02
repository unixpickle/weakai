package lstm

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/rnn"
)

// An Net is a single-layer recurrent neural net
// with LSTM hidden units.
type Net struct {
	Activation rnn.ActivationFunc `json:"-"`

	// The columns in this matrix correspond first
	// to input values and then to state values.
	OutWeights *linalg.Matrix

	// The elements in this vector correspond to
	// elements of the output.
	OutBiases linalg.Vector

	MemoryParams *LSTM

	currentState linalg.Vector
	inputs       []linalg.Vector
	lstmOutputs  []*LSTMOutput
	outputs      []linalg.Vector
}

func NewNet(activation rnn.ActivationFunc, inputSize, stateSize, outputSize int) *Net {
	return &Net{
		Activation:   activation,
		OutWeights:   linalg.NewMatrix(outputSize, inputSize+stateSize),
		OutBiases:    make(linalg.Vector, outputSize),
		MemoryParams: NewLSTM(inputSize, stateSize),
	}
}

func DeserializeNet(d []byte) (serializer.Serializer, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	} else if len(slice) != 2 {
		return nil, errors.New("invalid deserialized slice")
	}
	data, ok := slice[0].(serializer.Bytes)
	if !ok {
		return nil, errors.New("expected Bytes for first element")
	}
	activation, ok := slice[1].(rnn.ActivationFunc)
	if !ok {
		return nil, errors.New("expected ActivationFunc for second element")
	}
	var n Net
	if err := json.Unmarshal(data, &n); err != nil {
		return nil, err
	}
	n.Activation = activation
	return &n, nil
}

// Randomize randomly initializes the output
// and LSTM parameters.
func (r *Net) Randomize() {
	weightCoeff := math.Sqrt(3.0 / float64(r.OutWeights.Cols))
	for i := range r.OutWeights.Data {
		r.OutWeights.Data[i] = (rand.Float64()*2 - 1) * weightCoeff
	}
	r.MemoryParams.Randomize()
}

// StepTime gives the Net another input and
// returns the Net's output for that input.
func (r *Net) StepTime(in linalg.Vector) linalg.Vector {
	r.inputs = append(r.inputs, in)

	if r.currentState == nil {
		r.currentState = make(linalg.Vector, r.MemoryParams.StateSize)
	}

	output := r.MemoryParams.PropagateForward(r.currentState, in)
	r.currentState = output.NewState

	augmentedInput := &linalg.Matrix{
		Rows: len(in) + len(output.MaskedState),
		Cols: 1,
		Data: make([]float64, len(in)+len(output.MaskedState)),
	}
	copy(augmentedInput.Data, in)
	copy(augmentedInput.Data[len(in):], output.MaskedState)
	result := linalg.Vector(r.OutWeights.MulFast(augmentedInput).Data).Add(r.OutBiases)
	for i, x := range result {
		result[i] = r.Activation.Eval(x)
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
	grad := NewGradient(r.MemoryParams.InputSize, r.MemoryParams.StateSize, len(r.OutBiases),
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
	r.OutWeights.Add(g.OutWeights)
	r.OutBiases.Add(g.OutBiases)
	r.MemoryParams.InWeights.Add(g.InWeights)
	r.MemoryParams.InGate.Add(g.InGate)
	r.MemoryParams.RemGate.Add(g.RemGate)
	r.MemoryParams.OutGate.Add(g.OutGate)
	r.MemoryParams.InBiases.Add(g.InBiases)
	r.MemoryParams.InGateBiases.Add(g.InGateBiases)
	r.MemoryParams.RemGateBiases.Add(g.RemGateBiases)
	r.MemoryParams.OutGateBiases.Add(g.OutGateBiases)
	r.Reset()
}

// Reset goes back to time 0 and erases all the
// previous inputs and states.
// This does not reset weights or biases.
func (r *Net) Reset() {
	r.inputs = nil
	r.lstmOutputs = nil
	r.outputs = nil
	r.currentState = nil
}

func (r *Net) Alias() rnn.RNN {
	return &Net{
		Activation:   r.Activation,
		OutWeights:   r.OutWeights,
		OutBiases:    r.OutBiases,
		MemoryParams: r.MemoryParams,
	}
}

func (r *Net) Serialize() ([]byte, error) {
	jsonData, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	return serializer.SerializeSlice([]serializer.Serializer{
		serializer.Bytes(jsonData),
		r.Activation,
	})
}

func (r *Net) SerializerType() string {
	return serializerTypeNet
}

func (r *Net) computeShallowPartials(g *Gradient, costPartials []linalg.Vector) {
	inputCount := r.MemoryParams.InputSize

	for t, partial := range costPartials {
		for neuronIdx, costPartial := range partial {
			neuronOut := r.outputs[t][neuronIdx]
			activationPartial := r.Activation.Deriv(neuronOut)
			sumPartial := activationPartial * costPartial

			g.OutBiases[neuronIdx] += sumPartial

			inputGradList := g.InputGrads[t]
			for inputIdx, inputVal := range r.inputs[t] {
				val := g.OutWeights.Get(neuronIdx, inputIdx)
				val += inputVal * sumPartial
				g.OutWeights.Set(neuronIdx, inputIdx, val)

				weight := r.OutWeights.Get(neuronIdx, inputIdx)
				inputGradList[inputIdx] += weight * sumPartial
			}

			newStateList := r.lstmOutputs[t].NewState
			outputMaskList := r.lstmOutputs[t].OutputMask
			for hiddenIdx, maskedVal := range r.lstmOutputs[t].MaskedState {
				col := hiddenIdx + inputCount
				val := g.OutWeights.Get(neuronIdx, col)
				val += maskedVal * sumPartial
				g.OutWeights.Set(neuronIdx, col, val)

				weightVal := r.OutWeights.Get(neuronIdx, col)
				stateVal := newStateList[hiddenIdx]
				maskVal := outputMaskList[hiddenIdx]
				maskSigmoidPartial := (1 - maskVal) * maskVal
				maskSumPartial := maskSigmoidPartial * weightVal * stateVal * sumPartial
				g.OutGateBiases[hiddenIdx] += maskSumPartial
				for inputIdx1, inputVal := range r.inputs[t] {
					val := g.OutGate.Get(hiddenIdx, inputIdx1)
					val += inputVal * maskSumPartial
					g.OutGate.Set(hiddenIdx, inputIdx1, val)
				}
				if t > 0 {
					for hiddenIdx1, newStateVal := range r.lstmOutputs[t-1].NewState {
						col1 := hiddenIdx1 + inputCount
						val := g.OutGate.Get(hiddenIdx, col1)
						val += newStateVal * maskSumPartial
						g.OutGate.Set(hiddenIdx, col1, val)
					}
				}
			}
		}
	}
}

func (r *Net) computeDeepPartials(g *Gradient, costPartials []linalg.Vector) {
	hiddenCount := r.MemoryParams.StateSize
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
	hiddenCount := r.MemoryParams.StateSize
	inputCount := r.MemoryParams.InputSize
	current = make(linalg.Vector, hiddenCount)
	inGrad = make(linalg.Vector, inputCount)

	if t > 0 {
		older = make(linalg.Vector, hiddenCount)
	}

	for neuronIdx, partial := range costGrad {
		output := r.outputs[t][neuronIdx]
		activationDeriv := r.Activation.Deriv(output)
		sumDeriv := activationDeriv * partial
		for hiddenIdx, outMask := range r.lstmOutputs[t].OutputMask {
			col := hiddenIdx + inputCount
			weight := r.OutWeights.Get(neuronIdx, col)
			current[hiddenIdx] += outMask * weight * sumDeriv

			stateVal := r.lstmOutputs[t].NewState[hiddenIdx]
			maskSigmoidPartial := outMask * (1 - outMask)
			maskSumPartial := maskSigmoidPartial * stateVal * weight * sumDeriv
			for inputIdx := range inGrad {
				val := r.MemoryParams.OutGate.Get(hiddenIdx, inputIdx)
				inGrad[inputIdx] += val * maskSumPartial
			}

			if t == 0 {
				continue
			}

			for hiddenIdx1 := range older {
				col := inputCount + hiddenIdx1
				val := r.MemoryParams.OutGate.Get(hiddenIdx, col)
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

			weight := r.MemoryParams.RemGate.Get(hiddenIdx, inputIdx)
			g.InputGrads[t][inputIdx] += weight * maskSumPartial
		}

		for hiddenIdx1, inVal := range r.lstmOutputs[t-1].NewState {
			col := hiddenIdx1 + r.MemoryParams.InputSize
			val := g.RemGate.Get(hiddenIdx, col)
			val += inVal * maskSumPartial
			g.RemGate.Set(hiddenIdx, col, val)

			weight := r.MemoryParams.RemGate.Get(hiddenIdx, col)
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

			weight := r.MemoryParams.InGate.Get(hiddenIdx, inputIdx)
			g.InputGrads[t][inputIdx] += weight * maskSumPartial
		}

		if t == 0 {
			continue
		}

		for hiddenIdx1, inVal := range r.lstmOutputs[t-1].NewState {
			col := hiddenIdx1 + r.MemoryParams.InputSize
			val := g.InGate.Get(hiddenIdx, col)
			val += inVal * maskSumPartial
			g.InGate.Set(hiddenIdx, col, val)

			weight := r.MemoryParams.InGate.Get(hiddenIdx, col)
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

			weight := r.MemoryParams.InWeights.Get(hiddenIdx, inputIdx)
			g.InputGrads[t][inputIdx] += weight * inputSumPartial
		}

		if t == 0 {
			continue
		}

		for hiddenIdx1, inVal := range r.lstmOutputs[t-1].NewState {
			col := hiddenIdx1 + r.MemoryParams.InputSize
			val := g.InWeights.Get(hiddenIdx, col)
			val += inVal * inputSumPartial
			g.InWeights.Set(hiddenIdx, col, val)

			weight := r.MemoryParams.InWeights.Get(hiddenIdx, col)
			upstreamGrad[hiddenIdx1] += weight * inputSumPartial
		}
	}
}
