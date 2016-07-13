package rnn

import (
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

const initialRememberBias = 1

// LSTM is a Block that implements an LSTM unit.
type LSTM struct {
	hiddenSize int

	inputValue   *lstmGate
	inputGate    *lstmGate
	rememberGate *lstmGate
	outputGate   *lstmGate
}

// NewLSTM creates an LSTM with randomly initialized
// weights and biases.
// For each hidden unit, there are two elements of
// state, so the total state size is 2*hiddenSize.
func NewLSTM(inputSize, hiddenSize int) *LSTM {
	res := &LSTM{
		hiddenSize: hiddenSize,

		inputValue:   newLSTMGate(inputSize, hiddenSize, &neuralnet.HyperbolicTangent{}),
		inputGate:    newLSTMGate(inputSize, hiddenSize, &neuralnet.Sigmoid{}),
		rememberGate: newLSTMGate(inputSize, hiddenSize, &neuralnet.Sigmoid{}),
		outputGate:   newLSTMGate(inputSize, hiddenSize, &neuralnet.Sigmoid{}),
	}
	res.prioritizeRemembering()
	return res
}

// DeserializeLSTM creates an LSTM from some serialized
// data about the LSTM.
func DeserializeLSTM(d []byte) (*LSTM, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 5 {
		return nil, errors.New("invalid slice length in LSTM")
	}
	hiddenSize, ok := slice[0].(serializer.Int)
	inputValue, ok1 := slice[1].(*lstmGate)
	inputGate, ok2 := slice[2].(*lstmGate)
	rememberGate, ok3 := slice[3].(*lstmGate)
	outputGate, ok4 := slice[4].(*lstmGate)
	if !ok || !ok1 || !ok2 || !ok3 || !ok4 {
		return nil, errors.New("invalid types in LSTM slice")
	}
	return &LSTM{
		hiddenSize:   int(hiddenSize),
		inputValue:   inputValue,
		inputGate:    inputGate,
		rememberGate: rememberGate,
		outputGate:   outputGate,
	}, nil
}

// Parameters returns the LSTM's parameters in the
// following order: input weights, input biases,
// input gate weights, input gate biases, remember
// gate weights, remember gate biases, output gate
// weights, output gate biases.
func (l *LSTM) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{
		l.inputValue.Dense.Weights.Data,
		l.inputValue.Dense.Biases.Var,
		l.inputGate.Dense.Weights.Data,
		l.inputGate.Dense.Biases.Var,
		l.rememberGate.Dense.Weights.Data,
		l.rememberGate.Dense.Biases.Var,
		l.outputGate.Dense.Weights.Data,
		l.outputGate.Dense.Biases.Var,
	}
}

func (l *LSTM) StateSize() int {
	return l.hiddenSize * 2
}

func (l *LSTM) Batch(in *BlockInput) BlockOutput {
	n := len(in.Inputs)
	input := joinLSTMGateInputs(in)

	inValue := l.inputValue.Batch(input, n)
	inGate := l.inputGate.Batch(input, n)
	rememberGate := l.rememberGate.Batch(input, n)
	outputGate := l.outputGate.Batch(input, n)

	gatedIn := autofunc.Mul(inGate, inValue)
	gatedState := autofunc.Mul(rememberGate, joinLSTMInternalStates(in))
	newState := autofunc.Add(gatedIn, gatedState)

	// Pool the new state so that we do not
	// back propagate through it twice.
	newStateVar := &autofunc.Variable{Vector: newState.Output()}
	squashedOut := neuralnet.HyperbolicTangent{}.Apply(newStateVar)
	gatedOutput := autofunc.Mul(outputGate, squashedOut)

	return &lstmOutput{
		LaneCount:    n,
		OutStates:    newState,
		StatePool:    newStateVar,
		GatedOuts:    gatedOutput,
		WeavedStates: weaveLSTMOutputStates(newState.Output(), gatedOutput.Output(), n),
	}
}

func (l *LSTM) BatchR(v autofunc.RVector, in *BlockRInput) BlockROutput {
	n := len(in.Inputs)
	input := joinLSTMGateRInputs(in)

	inValue := l.inputValue.BatchR(v, input, n)
	inGate := l.inputGate.BatchR(v, input, n)
	rememberGate := l.rememberGate.BatchR(v, input, n)
	outputGate := l.outputGate.BatchR(v, input, n)

	gatedIn := autofunc.MulR(inGate, inValue)
	gatedState := autofunc.MulR(rememberGate, joinLSTMInternalRStates(in))
	newState := autofunc.AddR(gatedIn, gatedState)

	// Pool the new state so that we do not
	// back propagate through it twice.
	rawVar := &autofunc.Variable{Vector: newState.Output()}
	newStateVar := &autofunc.RVariable{
		Variable:   rawVar,
		ROutputVec: newState.ROutput(),
	}
	squashedOut := neuralnet.HyperbolicTangent{}.ApplyR(v, newStateVar)
	gatedOutput := autofunc.MulR(outputGate, squashedOut)

	return &lstmROutput{
		LaneCount: n,
		OutStates: newState,
		StatePool: newStateVar,
		GatedOuts: gatedOutput,

		WeavedStates:  weaveLSTMOutputStates(newState.Output(), gatedOutput.Output(), n),
		WeavedRStates: weaveLSTMOutputStates(newState.ROutput(), gatedOutput.ROutput(), n),
	}
}

func (l *LSTM) Serialize() ([]byte, error) {
	slist := []serializer.Serializer{
		serializer.Int(l.hiddenSize),
		l.inputValue,
		l.inputGate,
		l.rememberGate,
		l.outputGate,
	}
	return serializer.SerializeSlice(slist)
}

func (l *LSTM) SerializerType() string {
	return serializerTypeLSTM
}

func (l *LSTM) prioritizeRemembering() {
	rememberBiases := l.rememberGate.Dense.Biases.Var.Vector
	for i := range rememberBiases {
		rememberBiases[i] = initialRememberBias
	}
}

type lstmGate struct {
	Dense      *neuralnet.DenseLayer
	Activation neuralnet.Layer
}

func newLSTMGate(inputSize, hiddenSize int, activation neuralnet.Layer) *lstmGate {
	res := &lstmGate{
		Dense: &neuralnet.DenseLayer{
			InputCount:  inputSize + hiddenSize,
			OutputCount: hiddenSize,
		},
		Activation: activation,
	}
	res.Dense.Randomize()
	return res
}

func deserializeLSTMGate(d []byte) (*lstmGate, error) {
	list, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(list) != 2 {
		return nil, errors.New("invalid slice length for LSTM gate")
	}
	dense, ok := list[0].(*neuralnet.DenseLayer)
	activ, ok1 := list[1].(neuralnet.Layer)
	if !ok || !ok1 {
		return nil, errors.New("invalid types for list elements")
	}
	return &lstmGate{Dense: dense, Activation: activ}, nil
}

func (l *lstmGate) Batch(in autofunc.Result, n int) autofunc.Result {
	return l.Activation.Apply(l.Dense.Batch(in, n))
}

func (l *lstmGate) BatchR(v autofunc.RVector, in autofunc.RResult, n int) autofunc.RResult {
	return l.Activation.ApplyR(v, l.Dense.BatchR(v, in, n))
}

func (l *lstmGate) Serialize() ([]byte, error) {
	slist := []serializer.Serializer{l.Dense, l.Activation}
	return serializer.SerializeSlice(slist)
}

func (l *lstmGate) SerializerType() string {
	return serializerTypeLSTMGate
}

type lstmOutput struct {
	LaneCount    int
	OutStates    autofunc.Result
	StatePool    *autofunc.Variable
	GatedOuts    autofunc.Result
	WeavedStates linalg.Vector
}

func (l *lstmOutput) Outputs() []linalg.Vector {
	return splitVectors(l.GatedOuts.Output(), l.LaneCount)
}

func (l *lstmOutput) States() []linalg.Vector {
	return splitVectors(l.WeavedStates, l.LaneCount)
}

func (l *lstmOutput) Gradient(u *UpstreamGradient, g autofunc.Gradient) {
	var stateGrad linalg.Vector
	if u.States != nil {
		stateGrad = joinLSTMUpstreamInternalStates(u.States)
	} else {
		stateGrad = make(linalg.Vector, len(l.StatePool.Vector))
	}

	g[l.StatePool] = stateGrad
	outputGrad := addLSTMOutputGrads(u.Outputs, u.States)
	l.GatedOuts.PropagateGradient(outputGrad, g)
	delete(g, l.StatePool)

	l.OutStates.PropagateGradient(stateGrad, g)
}

type lstmROutput struct {
	LaneCount     int
	OutStates     autofunc.RResult
	StatePool     *autofunc.RVariable
	GatedOuts     autofunc.RResult
	WeavedStates  linalg.Vector
	WeavedRStates linalg.Vector
}

func (l *lstmROutput) Outputs() []linalg.Vector {
	return splitVectors(l.GatedOuts.Output(), l.LaneCount)
}

func (l *lstmROutput) ROutputs() []linalg.Vector {
	return splitVectors(l.GatedOuts.ROutput(), l.LaneCount)
}

func (l *lstmROutput) States() []linalg.Vector {
	return splitVectors(l.WeavedStates, l.LaneCount)
}

func (l *lstmROutput) RStates() []linalg.Vector {
	return splitVectors(l.WeavedRStates, l.LaneCount)
}

func (l *lstmROutput) RGradient(u *UpstreamRGradient, rg autofunc.RGradient,
	g autofunc.Gradient) {
	// The gradient is used for temporary values.
	if g == nil {
		g = autofunc.Gradient{}
	}

	var stateGrad, stateRGrad linalg.Vector
	if u.States != nil {
		stateGrad = joinLSTMUpstreamInternalStates(u.States)
		stateRGrad = joinLSTMUpstreamInternalStates(u.RStates)
	} else {
		stateGrad = make(linalg.Vector, len(l.StatePool.Output()))
		stateRGrad = make(linalg.Vector, len(l.StatePool.ROutput()))
	}

	g[l.StatePool.Variable] = stateGrad
	rg[l.StatePool.Variable] = stateRGrad
	l.GatedOuts.PropagateRGradient(addLSTMOutputGrads(u.Outputs, u.States),
		addLSTMOutputGrads(u.ROutputs, u.RStates), rg, g)
	delete(g, l.StatePool.Variable)
	delete(rg, l.StatePool.Variable)

	l.OutStates.PropagateRGradient(stateGrad, stateRGrad, rg, g)
}

func joinLSTMGateInputs(in *BlockInput) autofunc.Result {
	results := make([]autofunc.Result, 0, len(in.States)*2)
	for i, fullState := range in.States {
		outputState := autofunc.Slice(fullState, 0, len(fullState.Vector)/2)
		results = append(results, in.Inputs[i], outputState)
	}
	return autofunc.Concat(results...)
}

func joinLSTMGateRInputs(in *BlockRInput) autofunc.RResult {
	results := make([]autofunc.RResult, 0, len(in.States)*2)
	for i, fullState := range in.States {
		outputState := autofunc.SliceR(fullState, 0, len(fullState.Variable.Vector)/2)
		results = append(results, in.Inputs[i], outputState)
	}
	return autofunc.ConcatR(results...)
}

func joinLSTMInternalStates(in *BlockInput) autofunc.Result {
	results := make([]autofunc.Result, 0, len(in.States))
	for _, fullState := range in.States {
		startIdx := len(fullState.Vector) / 2
		outputState := autofunc.Slice(fullState, startIdx, startIdx*2)
		results = append(results, outputState)
	}
	return autofunc.Concat(results...)
}

func joinLSTMInternalRStates(in *BlockRInput) autofunc.RResult {
	results := make([]autofunc.RResult, 0, len(in.States))
	for _, fullState := range in.States {
		startIdx := len(fullState.Variable.Vector) / 2
		outputState := autofunc.SliceR(fullState, startIdx, startIdx*2)
		results = append(results, outputState)
	}
	return autofunc.ConcatR(results...)
}

func joinLSTMUpstreamInternalStates(vecs []linalg.Vector) linalg.Vector {
	stateLen := len(vecs[0]) / 2
	res := make(linalg.Vector, stateLen*len(vecs))
	for i, fullState := range vecs {
		halfIdx := len(fullState) / 2
		copy(res[i*stateLen:(i+1)*stateLen], fullState[halfIdx:2*halfIdx])
	}
	return res
}

func weaveLSTMOutputStates(states, outputs linalg.Vector, n int) linalg.Vector {
	stateLen := len(states) / n
	res := make(linalg.Vector, len(states)*2)
	for i := 0; i < n; i++ {
		startIdx := i * stateLen
		endIdx := (i + 1) * stateLen
		copy(res[startIdx*2:], outputs[startIdx:endIdx])
		copy(res[startIdx*2+stateLen:], states[startIdx:endIdx])
	}
	return res
}

func addLSTMOutputGrads(outputs, states []linalg.Vector) linalg.Vector {
	var stateLen, stateCount int
	if outputs != nil {
		stateLen = len(outputs[0])
		stateCount = len(outputs)
	} else {
		stateLen = len(states[0]) / 2
		stateCount = len(states)
	}
	res := make(linalg.Vector, stateLen*stateCount)
	for i := 0; i < stateCount; i++ {
		outputIdx := i * stateLen
		subVec := res[outputIdx : outputIdx+stateLen]
		if outputs != nil {
			copy(subVec, outputs[i])
			if states != nil {
				subVec.Add(states[i][:stateLen])
			}
		} else {
			copy(subVec, states[i])
		}
	}
	return res
}
