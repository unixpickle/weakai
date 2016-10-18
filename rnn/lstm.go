package rnn

import (
	"encoding/json"
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

const initialRememberBias = 1

func init() {
	var l LSTM
	serializer.RegisterTypedDeserializer(l.SerializerType(), DeserializeLSTM)

	var lg lstmGate
	serializer.RegisterTypedDeserializer(lg.SerializerType(), deserializeLSTMGate)
}

// LSTM is a Block that implements an LSTM unit.
type LSTM struct {
	hiddenSize int

	inputValue   *lstmGate
	inputGate    *lstmGate
	rememberGate *lstmGate
	outputGate   *lstmGate
	initState    *autofunc.Variable
}

// NewLSTM creates an LSTM with randomly initialized
// weights and biases.
// For each hidden unit, there are two elements of
// state, so the total state size is 2*hiddenSize.
func NewLSTM(inputSize, hiddenSize int) *LSTM {
	htan := &neuralnet.HyperbolicTangent{}
	sigmoid := &neuralnet.Sigmoid{}
	res := &LSTM{
		hiddenSize:   hiddenSize,
		inputValue:   newLSTMGate(inputSize, hiddenSize*2, hiddenSize, htan),
		inputGate:    newLSTMGate(inputSize, hiddenSize*2, hiddenSize, sigmoid),
		rememberGate: newLSTMGate(inputSize, hiddenSize*2, hiddenSize, sigmoid),
		outputGate:   newLSTMGate(inputSize, hiddenSize*2, hiddenSize, sigmoid),
		initState:    &autofunc.Variable{Vector: make(linalg.Vector, hiddenSize*2)},
	}
	res.prioritizeRemembering()
	return res
}

// DeserializeLSTM deserializes an LSTM.
func DeserializeLSTM(d []byte) (*LSTM, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 6 {
		return nil, errors.New("invalid slice length in LSTM")
	}
	hiddenSize, ok := slice[0].(serializer.Int)
	inputValue, ok1 := slice[1].(*lstmGate)
	inputGate, ok2 := slice[2].(*lstmGate)
	rememberGate, ok3 := slice[3].(*lstmGate)
	outputGate, ok4 := slice[4].(*lstmGate)
	initStateData, ok5 := slice[5].(serializer.Bytes)
	if !ok || !ok1 || !ok2 || !ok3 || !ok4 || !ok5 {
		return nil, errors.New("invalid types in LSTM slice")
	}
	var initState autofunc.Variable
	if err := json.Unmarshal(initStateData, &initState); err != nil {
		return nil, err
	}
	return &LSTM{
		hiddenSize:   int(hiddenSize),
		inputValue:   inputValue,
		inputGate:    inputGate,
		rememberGate: rememberGate,
		outputGate:   outputGate,
		initState:    &initState,
	}, nil
}

// StartState returns the trainable start state.
func (l *LSTM) StartState() State {
	return lstmState{
		Internal: l.initState.Vector[:len(l.initState.Vector)/2],
		Output:   l.initState.Vector[len(l.initState.Vector)/2:],
	}
}

// StartRState is like StartState but with an RState.
func (l *LSTM) StartRState(rv autofunc.RVector) RState {
	rVar := autofunc.NewRVariable(l.initState, rv)
	return lstmRState{
		Internal:  l.initState.Vector[:len(l.initState.Vector)/2],
		InternalR: rVar.ROutputVec[:len(l.initState.Vector)/2],
		Output:    l.initState.Vector[len(l.initState.Vector)/2:],
		OutputR:   rVar.ROutputVec[len(l.initState.Vector)/2:],
	}
}

// PropagateStart performs back-propagation through the
// start state.
func (l *LSTM) PropagateStart(s []StateGrad, g autofunc.Gradient) {
	if vec, ok := g[l.initState]; ok {
		for _, x := range s {
			vec[:len(vec)/2].Add(linalg.Vector(x.(lstmState).Internal))
			vec[len(vec)/2:].Add(linalg.Vector(x.(lstmState).Output))
		}
	}
}

// PropagateStartR is like PropagateStart but with
// RStateGrads.
func (l *LSTM) PropagateStartR(s []RStateGrad, rg autofunc.RGradient, g autofunc.Gradient) {
	if g != nil {
		if vec, ok := g[l.initState]; ok {
			for _, x := range s {
				vec[:len(vec)/2].Add(x.(lstmRState).Internal)
				vec[len(vec)/2:].Add(x.(lstmRState).Output)
			}
		}
	}
	if vec, ok := rg[l.initState]; ok {
		for _, x := range s {
			vec[:len(vec)/2].Add(x.(lstmRState).InternalR)
			vec[len(vec)/2:].Add(x.(lstmRState).OutputR)
		}
	}
}

// ApplyBlock applies the LSTM to a batch of inputs.
func (l *LSTM) ApplyBlock(s []State, in []autofunc.Result) BlockResult {
	var internalPool, lastOutPool []*autofunc.Variable
	res := autofunc.PoolAll(in, func(in []autofunc.Result) autofunc.Result {
		var weavedInputs []autofunc.Result
		var internalResults []autofunc.Result
		for i, sObj := range s {
			state := sObj.(lstmState)
			internalVar := &autofunc.Variable{Vector: state.Internal}
			lastOutVar := &autofunc.Variable{Vector: state.Output}

			internalPool = append(internalPool, internalVar)
			lastOutPool = append(lastOutPool, lastOutVar)

			weavedInputs = append(weavedInputs, in[i], internalVar, lastOutVar)
			internalResults = append(internalResults, internalVar)
		}

		gateIn := autofunc.Concat(weavedInputs...)
		inValue := l.inputValue.Batch(gateIn, len(in))
		inGate := l.inputGate.Batch(gateIn, len(in))
		rememberGate := l.rememberGate.Batch(gateIn, len(in))

		lastState := autofunc.Concat(internalResults...)
		newState := autofunc.Add(autofunc.Mul(rememberGate, lastState),
			autofunc.Mul(inValue, inGate))

		return autofunc.Pool(newState, func(newState autofunc.Result) autofunc.Result {
			var newWeaved []autofunc.Result
			for i, state := range autofunc.Split(len(in), newState) {
				newWeaved = append(newWeaved, in[i], state, lastOutPool[i])
			}
			newGateIn := autofunc.Concat(newWeaved...)
			outGate := l.outputGate.Batch(newGateIn, len(in))
			outValues := neuralnet.HyperbolicTangent{}.Apply(newState)
			return autofunc.Concat(newState, autofunc.Mul(outGate, outValues))
		})
	})

	states, outs := splitLSTMOutput(len(in), res.Output())
	return &lstmResult{
		CellStates:   states,
		OutputVecs:   outs,
		InternalPool: internalPool,
		LastOutPool:  lastOutPool,
		JoinedOut:    res,
	}
}

// ApplyBlockR is like ApplyBlock, but with support for
// the R operator.
func (l *LSTM) ApplyBlockR(rv autofunc.RVector, s []RState, in []autofunc.RResult) BlockRResult {
	var internalPool, lastOutPool []*autofunc.Variable
	res := autofunc.PoolAllR(in, func(in []autofunc.RResult) autofunc.RResult {
		var lastOutPoolR []*autofunc.RVariable
		var weavedInputs []autofunc.RResult
		var internalResults []autofunc.RResult
		for i, sObj := range s {
			state := sObj.(lstmRState)
			internalVar := &autofunc.Variable{Vector: state.Internal}
			lastOutVar := &autofunc.Variable{Vector: state.Output}

			internalPool = append(internalPool, internalVar)
			lastOutPool = append(lastOutPool, lastOutVar)

			internalR := &autofunc.RVariable{
				Variable:   internalVar,
				ROutputVec: state.InternalR,
			}
			lastOutR := &autofunc.RVariable{
				Variable:   lastOutVar,
				ROutputVec: state.OutputR,
			}

			lastOutPoolR = append(lastOutPoolR, lastOutR)
			weavedInputs = append(weavedInputs, in[i], internalR, lastOutR)
			internalResults = append(internalResults, internalR)
		}

		gateIn := autofunc.ConcatR(weavedInputs...)
		inValue := l.inputValue.BatchR(rv, gateIn, len(in))
		inGate := l.inputGate.BatchR(rv, gateIn, len(in))
		rememberGate := l.rememberGate.BatchR(rv, gateIn, len(in))

		lastState := autofunc.ConcatR(internalResults...)
		newState := autofunc.AddR(autofunc.MulR(rememberGate, lastState),
			autofunc.MulR(inValue, inGate))

		return autofunc.PoolR(newState, func(newState autofunc.RResult) autofunc.RResult {
			var newWeaved []autofunc.RResult
			for i, state := range autofunc.SplitR(len(in), newState) {
				newWeaved = append(newWeaved, in[i], state, lastOutPoolR[i])
			}
			newGateIn := autofunc.ConcatR(newWeaved...)
			outGate := l.outputGate.BatchR(rv, newGateIn, len(in))
			outValues := neuralnet.HyperbolicTangent{}.ApplyR(rv, newState)
			return autofunc.ConcatR(newState, autofunc.MulR(outGate, outValues))
		})
	})

	states, outs := splitLSTMOutput(len(in), res.Output())
	statesR, outsR := splitLSTMOutput(len(in), res.ROutput())
	return &lstmRResult{
		CellStates:   states,
		RCellStates:  statesR,
		OutputVecs:   outs,
		ROutputVecs:  outsR,
		InternalPool: internalPool,
		LastOutPool:  lastOutPool,
		JoinedOut:    res,
	}
}

// Parameters returns the LSTM's parameters in the
// following order: input weights, input biases,
// input gate weights, input gate biases, remember
// gate weights, remember gate biases, output gate
// weights, output gate biases, init state biases.
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
		l.initState,
	}
}

// SerializerType returns the unique ID used to serialize
// an LSTM with the serializer package.
func (l *LSTM) SerializerType() string {
	return "github.com/unixpickle/weakai/rnn.LSTM"
}

// Serialize serializes the LSTM.
func (l *LSTM) Serialize() ([]byte, error) {
	initData, err := json.Marshal(l.initState)
	if err != nil {
		return nil, err
	}
	slist := []serializer.Serializer{
		serializer.Int(l.hiddenSize),
		l.inputValue,
		l.inputGate,
		l.rememberGate,
		l.outputGate,
		serializer.Bytes(initData),
	}
	return serializer.SerializeSlice(slist)
}

func (l *LSTM) prioritizeRemembering() {
	rememberBiases := l.rememberGate.Dense.Biases.Var.Vector
	for i := range rememberBiases {
		rememberBiases[i] = initialRememberBias
	}
}

type lstmState struct {
	Output   linalg.Vector
	Internal linalg.Vector
}

type lstmRState struct {
	Output    linalg.Vector
	OutputR   linalg.Vector
	Internal  linalg.Vector
	InternalR linalg.Vector
}

type lstmGate struct {
	Dense      *neuralnet.DenseLayer
	Activation neuralnet.Layer
}

func newLSTMGate(inputSize, inHidden, outHidden int, activation neuralnet.Layer) *lstmGate {
	res := &lstmGate{
		Dense: &neuralnet.DenseLayer{
			InputCount:  inputSize + inHidden,
			OutputCount: outHidden,
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
	return "github.com/unixpickle/weakai/rnn.lstmGate"
}

type lstmResult struct {
	CellStates []linalg.Vector
	OutputVecs []linalg.Vector

	InternalPool []*autofunc.Variable
	LastOutPool  []*autofunc.Variable

	JoinedOut autofunc.Result
}

func (l *lstmResult) Outputs() []linalg.Vector {
	return l.OutputVecs
}

func (l *lstmResult) States() []State {
	res := make([]State, len(l.OutputVecs))
	for i, x := range l.CellStates {
		res[i] = lstmState{Internal: x, Output: l.OutputVecs[i]}
	}
	return res
}

func (l *lstmResult) PropagateGradient(u []linalg.Vector, s []StateGrad,
	g autofunc.Gradient) []StateGrad {
	n := len(l.OutputVecs)
	if n == 0 || (u == nil && s == nil) {
		return nil
	}
	cellCount := len(l.OutputVecs[0])

	downstream := make(linalg.Vector, n*cellCount*2)
	for i, x := range u {
		downstream[(i+n)*cellCount : (i+n+1)*cellCount].Add(x)
	}
	for i, stateObj := range s {
		if stateObj != nil {
			state := stateObj.(lstmState)
			downstream[i*cellCount : (i+1)*cellCount].Add(state.Internal)
			downstream[(i+n)*cellCount : (i+n+1)*cellCount].Add(state.Output)
		}
	}

	for _, vars := range [][]*autofunc.Variable{l.InternalPool, l.LastOutPool} {
		for _, v := range vars {
			g[v] = make(linalg.Vector, len(v.Vector))
		}
	}
	l.JoinedOut.PropagateGradient(downstream, g)

	outGrad := make([]StateGrad, n)
	for i := range outGrad {
		outGrad[i] = lstmState{
			Internal: g[l.InternalPool[i]],
			Output:   g[l.LastOutPool[i]],
		}
		delete(g, l.InternalPool[i])
		delete(g, l.LastOutPool[i])
	}

	return outGrad
}

type lstmRResult struct {
	CellStates  []linalg.Vector
	RCellStates []linalg.Vector
	OutputVecs  []linalg.Vector
	ROutputVecs []linalg.Vector

	InternalPool []*autofunc.Variable
	LastOutPool  []*autofunc.Variable

	JoinedOut autofunc.RResult
}

func (l *lstmRResult) Outputs() []linalg.Vector {
	return l.OutputVecs
}

func (l *lstmRResult) ROutputs() []linalg.Vector {
	return l.ROutputVecs
}

func (l *lstmRResult) RStates() []RState {
	res := make([]RState, len(l.OutputVecs))
	for i, x := range l.CellStates {
		res[i] = lstmRState{
			Internal:  x,
			Output:    l.OutputVecs[i],
			InternalR: l.RCellStates[i],
			OutputR:   l.ROutputVecs[i],
		}
	}
	return res
}

func (l *lstmRResult) PropagateRGradient(u, uR []linalg.Vector, s []RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) []RStateGrad {
	n := len(l.OutputVecs)
	if n == 0 || (u == nil && s == nil) {
		return nil
	}
	cellCount := len(l.OutputVecs[0])

	if g == nil {
		g = autofunc.Gradient{}
	}

	downstream := make(linalg.Vector, n*cellCount*2)
	downstreamR := make(linalg.Vector, n*cellCount*2)
	for i, x := range u {
		downstream[(i+n)*cellCount : (i+n+1)*cellCount].Add(x)
		downstreamR[(i+n)*cellCount : (i+n+1)*cellCount].Add(uR[i])
	}
	for i, stateObj := range s {
		if stateObj != nil {
			state := stateObj.(lstmRState)
			downstream[i*cellCount : (i+1)*cellCount].Add(state.Internal)
			downstream[(i+n)*cellCount : (i+n+1)*cellCount].Add(state.Output)
			downstreamR[i*cellCount : (i+1)*cellCount].Add(state.InternalR)
			downstreamR[(i+n)*cellCount : (i+n+1)*cellCount].Add(state.OutputR)
		}
	}

	for _, vars := range [][]*autofunc.Variable{l.InternalPool, l.LastOutPool} {
		for _, v := range vars {
			g[v] = make(linalg.Vector, len(v.Vector))
			rg[v] = make(linalg.Vector, len(v.Vector))
		}
	}
	l.JoinedOut.PropagateRGradient(downstream, downstreamR, rg, g)

	outGrad := make([]RStateGrad, n)
	for i := range outGrad {
		outGrad[i] = lstmRState{
			Internal:  g[l.InternalPool[i]],
			Output:    g[l.LastOutPool[i]],
			InternalR: rg[l.InternalPool[i]],
			OutputR:   rg[l.LastOutPool[i]],
		}
		delete(g, l.InternalPool[i])
		delete(g, l.LastOutPool[i])
		delete(rg, l.InternalPool[i])
		delete(rg, l.LastOutPool[i])
	}

	return outGrad
}

func splitLSTMOutput(n int, statesAndOuts linalg.Vector) (states, outs []linalg.Vector) {
	cellCount := len(statesAndOuts) / (2 * n)
	for i := 0; i < n; i++ {
		states = append(states, statesAndOuts[i*cellCount:(i+1)*cellCount])
		outs = append(outs, statesAndOuts[(i+n)*cellCount:(i+n+1)*cellCount])
	}
	return
}
