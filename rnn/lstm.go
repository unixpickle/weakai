package rnn

import (
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

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
func NewLSTM(inputSize, hiddenSize int) *LSTM {
	return &LSTM{
		hiddenSize: hiddenSize,

		inputValue:   newLSTMGate(inputSize, hiddenSize, &neuralnet.HyperbolicTangent{}),
		inputGate:    newLSTMGate(inputSize, hiddenSize, &neuralnet.Sigmoid{}),
		rememberGate: newLSTMGate(inputSize, hiddenSize, &neuralnet.Sigmoid{}),
		outputGate:   newLSTMGate(inputSize, hiddenSize, &neuralnet.Sigmoid{}),
	}
}

// DeserializeLSTM creates an LSTM from some serialized
// data about the LSTM.
func DeserializeLSTM(d []byte) (serializer.Serializer, error) {
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
	return l.hiddenSize
}

func (l *LSTM) Batch(in *BlockInput) BlockOutput {
	n := len(in.Inputs)
	input := joinBlockInput(in)

	inValue := l.inputValue.Batch(input, n)
	inGate := l.inputGate.Batch(input, n)
	rememberGate := l.rememberGate.Batch(input, n)
	outputGate := l.outputGate.Batch(input, n)

	gatedIn := autofunc.Mul(inGate, inValue)
	gatedState := autofunc.Mul(rememberGate, joinVariables(in.States))
	newState := autofunc.Add(gatedIn, gatedState)

	// Pool the new state so that we do not
	// back propagate through it twice.
	newStateVar := &autofunc.Variable{Vector: newState.Output()}
	gatedOutput := autofunc.Mul(outputGate, newStateVar)

	return &lstmOutput{
		LaneCount: n,
		OutStates: newState,
		StatePool: newStateVar,
		GatedOuts: gatedOutput,
	}
}

func (l *LSTM) BatchR(v autofunc.RVector, in *BlockRInput) BlockROutput {
	n := len(in.Inputs)
	input := joinBlockRInput(in)

	inValue := l.inputValue.BatchR(v, input, n)
	inGate := l.inputGate.BatchR(v, input, n)
	rememberGate := l.rememberGate.BatchR(v, input, n)
	outputGate := l.outputGate.BatchR(v, input, n)

	gatedIn := autofunc.MulR(inGate, inValue)
	gatedState := autofunc.MulR(rememberGate, joinRVariables(in.States))
	newState := autofunc.AddR(gatedIn, gatedState)

	// Pool the new state so that we do not
	// back propagate through it twice.
	rawVar := &autofunc.Variable{Vector: newState.Output()}
	newStateVar := &autofunc.RVariable{
		Variable:   rawVar,
		ROutputVec: newState.ROutput(),
	}
	gatedOutput := autofunc.MulR(outputGate, newStateVar)

	return &lstmROutput{
		LaneCount: n,
		OutStates: newState,
		StatePool: newStateVar,
		GatedOuts: gatedOutput,
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

func deserializeLSTMGate(d []byte) (serializer.Serializer, error) {
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
	LaneCount int
	OutStates autofunc.Result
	StatePool *autofunc.Variable
	GatedOuts autofunc.Result
}

func (l *lstmOutput) Outputs() []linalg.Vector {
	return splitVectors(l.GatedOuts.Output(), l.LaneCount)
}

func (l *lstmOutput) States() []linalg.Vector {
	return splitVectors(l.StatePool.Vector, l.LaneCount)
}

func (l *lstmOutput) Gradient(u *UpstreamGradient, g autofunc.Gradient) {
	stateGrad := make(linalg.Vector, len(l.StatePool.Vector))
	if u.States != nil {
		joinVectorsInPlace(stateGrad, u.States)
	}
	if u.Outputs != nil {
		g[l.StatePool] = stateGrad
		l.GatedOuts.PropagateGradient(joinVectors(u.Outputs), g)
		delete(g, l.StatePool)
	}
	l.OutStates.PropagateGradient(stateGrad, g)
}

type lstmROutput struct {
	LaneCount int
	OutStates autofunc.RResult
	StatePool *autofunc.RVariable
	GatedOuts autofunc.RResult
}

func (l *lstmROutput) Outputs() []linalg.Vector {
	return splitVectors(l.GatedOuts.Output(), l.LaneCount)
}

func (l *lstmROutput) ROutputs() []linalg.Vector {
	return splitVectors(l.GatedOuts.ROutput(), l.LaneCount)
}

func (l *lstmROutput) States() []linalg.Vector {
	return splitVectors(l.StatePool.Output(), l.LaneCount)
}

func (l *lstmROutput) RStates() []linalg.Vector {
	return splitVectors(l.StatePool.ROutput(), l.LaneCount)
}

func (l *lstmROutput) RGradient(u *UpstreamRGradient, rg autofunc.RGradient,
	g autofunc.Gradient) {
	// The gradient is used for temporary values.
	if g == nil {
		g = autofunc.Gradient{}
	}

	stateGrad := make(linalg.Vector, len(l.StatePool.Output()))
	stateRGrad := make(linalg.Vector, len(l.StatePool.ROutput()))
	if u.States != nil {
		joinVectorsInPlace(stateGrad, u.States)
		joinVectorsInPlace(stateRGrad, u.RStates)
	}
	if u.Outputs != nil {
		g[l.StatePool.Variable] = stateGrad
		rg[l.StatePool.Variable] = stateRGrad
		l.GatedOuts.PropagateRGradient(joinVectors(u.Outputs),
			joinVectors(u.ROutputs), rg, g)
		delete(g, l.StatePool.Variable)
		delete(rg, l.StatePool.Variable)
	}
	l.OutStates.PropagateRGradient(stateGrad, stateRGrad, rg, g)
}
