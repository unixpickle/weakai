package rnn

import (
	"encoding/json"
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var g GRU
	serializer.RegisterTypedDeserializer(g.SerializerType(), DeserializeGRU)
}

// GRU is a Block that implements an GRU unit, as
// defined in http://arxiv.org/pdf/1406.1078v3.pdf.
type GRU struct {
	hiddenSize int
	inputValue *lstmGate
	resetGate  *lstmGate
	updateGate *lstmGate
	initState  *autofunc.Variable
}

// NewGRU creates a GRU with randomly initialized
// weights and biases.
func NewGRU(inputSize, hiddenSize int) *GRU {
	htan := &neuralnet.HyperbolicTangent{}
	sigmoid := &neuralnet.Sigmoid{}
	res := &GRU{
		hiddenSize: hiddenSize,
		inputValue: newLSTMGate(inputSize, hiddenSize, hiddenSize, htan),
		resetGate:  newLSTMGate(inputSize, hiddenSize, hiddenSize, sigmoid),
		updateGate: newLSTMGate(inputSize, hiddenSize, hiddenSize, sigmoid),
		initState:  &autofunc.Variable{Vector: make(linalg.Vector, hiddenSize)},
	}
	return res
}

// DeserializeGRU creates a GRU from some serialized
// data about the GRU.
func DeserializeGRU(d []byte) (*GRU, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 5 {
		return nil, errors.New("invalid slice length in GRU")
	}
	hiddenSize, ok := slice[0].(serializer.Int)
	inputValue, ok1 := slice[1].(*lstmGate)
	resetGate, ok2 := slice[2].(*lstmGate)
	updateGate, ok3 := slice[3].(*lstmGate)
	initStateData, ok4 := slice[4].(serializer.Bytes)
	if !ok || !ok1 || !ok2 || !ok3 || !ok4 {
		return nil, errors.New("invalid types in GRU slice")
	}
	var initState autofunc.Variable
	if err := json.Unmarshal(initStateData, &initState); err != nil {
		return nil, errors.New("invalid init state in GRU slice")
	}
	return &GRU{
		hiddenSize: int(hiddenSize),
		inputValue: inputValue,
		resetGate:  resetGate,
		updateGate: updateGate,
		initState:  &initState,
	}, nil
}

// StartState returns the trainable start state.
func (g *GRU) StartState() State {
	return VecState(g.initState.Vector)
}

// StartStateR is like StartState but with r-operators.
func (g *GRU) StartRState(rv autofunc.RVector) RState {
	resVar := autofunc.NewRVariable(g.initState, rv)
	return VecRState{State: resVar.Output(), RState: resVar.ROutput()}
}

// PropagateStart propagates through the start state.
func (g *GRU) PropagateStart(s []StateGrad, grad autofunc.Gradient) {
	vec, ok := grad[g.initState]
	if !ok {
		return
	}
	for _, x := range s {
		vec.Add(linalg.Vector(x.(VecStateGrad)))
	}
}

// PropagateStartR propagates through the start state.
func (g *GRU) PropagateStartR(s []RStateGrad, rg autofunc.RGradient, grad autofunc.Gradient) {
	if grad != nil {
		if vec, ok := grad[g.initState]; ok {
			for _, x := range s {
				vec.Add(x.(VecRStateGrad).State)
			}
		}
	}
	if vec, ok := rg[g.initState]; ok {
		for _, x := range s {
			vec.Add(x.(VecRStateGrad).RState)
		}
	}
}

// ApplyBlock applies the block to an input.
func (g *GRU) ApplyBlock(s []State, in []autofunc.Result) BlockResult {
	var stateVars []*autofunc.Variable
	var stateRes []autofunc.Result
	var gateInputs []autofunc.Result
	for i, x := range s {
		stateVar := &autofunc.Variable{Vector: linalg.Vector(x.(VecState))}
		stateVars = append(stateVars, stateVar)
		stateRes = append(stateRes, stateVar)
		gateInputs = append(gateInputs, in[i], stateVar)
	}
	n := len(in)

	gateInput := autofunc.Concat(gateInputs...)
	stateIn := autofunc.Concat(stateRes...)

	resetMask := g.resetGate.Batch(gateInput, n)
	updateMask := g.updateGate.Batch(gateInput, n)

	maskedByReset := autofunc.Mul(resetMask, stateIn)
	inputValue := autofunc.PoolSplit(n, maskedByReset,
		func(newStates []autofunc.Result) autofunc.Result {
			var newGateInputs []autofunc.Result
			for i, input := range in {
				newGateInputs = append(newGateInputs, input, newStates[i])
			}
			newIn := autofunc.Concat(newGateInputs...)
			return g.inputValue.Batch(newIn, n)
		})

	newState := autofunc.Pool(updateMask, func(umask autofunc.Result) autofunc.Result {
		updateComplement := autofunc.AddScaler(autofunc.Scale(umask, -1), 1)
		return autofunc.Add(autofunc.Mul(umask, stateIn),
			autofunc.Mul(updateComplement, inputValue))
	})

	return &gruResult{
		InStates: stateVars,
		Output:   newState,
	}
}

// ApplyBlockR applies the block to an input.
func (g *GRU) ApplyBlockR(rv autofunc.RVector, s []RState, in []autofunc.RResult) BlockRResult {
	var stateVars []*autofunc.Variable
	var stateRes []autofunc.RResult
	var gateInputs []autofunc.RResult
	for i, x := range s {
		stateVar := &autofunc.Variable{Vector: linalg.Vector(x.(VecRState).State)}
		stateVars = append(stateVars, stateVar)

		stateVarR := &autofunc.RVariable{
			Variable:   stateVar,
			ROutputVec: x.(VecRState).RState,
		}
		stateRes = append(stateRes, stateVarR)
		gateInputs = append(gateInputs, in[i], stateVarR)
	}
	n := len(in)

	gateInput := autofunc.ConcatR(gateInputs...)
	stateIn := autofunc.ConcatR(stateRes...)

	resetMask := g.resetGate.BatchR(rv, gateInput, n)
	updateMask := g.updateGate.BatchR(rv, gateInput, n)

	maskedByReset := autofunc.MulR(resetMask, stateIn)
	inputValue := autofunc.PoolSplitR(n, maskedByReset,
		func(newStates []autofunc.RResult) autofunc.RResult {
			var newGateInputs []autofunc.RResult
			for i, input := range in {
				newGateInputs = append(newGateInputs, input, newStates[i])
			}
			newIn := autofunc.ConcatR(newGateInputs...)
			return g.inputValue.BatchR(rv, newIn, n)
		})

	newState := autofunc.PoolR(updateMask, func(umask autofunc.RResult) autofunc.RResult {
		updateComplement := autofunc.AddScalerR(autofunc.ScaleR(umask, -1), 1)
		return autofunc.AddR(autofunc.MulR(umask, stateIn),
			autofunc.MulR(updateComplement, inputValue))
	})

	return &gruRResult{
		InStates: stateVars,
		Output:   newState,
	}
}

// Parameters returns the GRU's parameters in the
// following order: input weights, input biases,
// reset gate weights, reset gate biases, update
// gate weights, update gate biases, initial state
// biases.
func (g *GRU) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{
		g.inputValue.Dense.Weights.Data,
		g.inputValue.Dense.Biases.Var,
		g.resetGate.Dense.Weights.Data,
		g.resetGate.Dense.Biases.Var,
		g.updateGate.Dense.Weights.Data,
		g.updateGate.Dense.Biases.Var,
		g.initState,
	}
}

// Serialize serializes the block.
func (g *GRU) Serialize() ([]byte, error) {
	initData, err := json.Marshal(g.initState)
	if err != nil {
		return nil, err
	}
	slist := []serializer.Serializer{
		serializer.Int(g.hiddenSize),
		g.inputValue,
		g.resetGate,
		g.updateGate,
		serializer.Bytes(initData),
	}
	return serializer.SerializeSlice(slist)
}

// SerializerType returns the unique ID used to serialize
// a GRU with the serializer package.
func (g *GRU) SerializerType() string {
	return "github.com/unixpickle/weakai/rnn.GRU"
}

type gruResult struct {
	InStates []*autofunc.Variable
	Output   autofunc.Result
}

func (g *gruResult) Outputs() []linalg.Vector {
	return splitVectors(g.Output.Output(), len(g.InStates))
}

func (g *gruResult) States() []State {
	var res []State
	for _, stateVec := range g.Outputs() {
		res = append(res, VecState(stateVec))
	}
	return res
}

func (g *gruResult) PropagateGradient(u []linalg.Vector, s []StateGrad,
	grad autofunc.Gradient) []StateGrad {
	if len(g.InStates) == 0 {
		return nil
	}
	downstream := make(linalg.Vector, len(g.Output.Output()))
	cells := len(downstream) / len(g.InStates)
	for i, stateObj := range s {
		if stateObj != nil {
			state := linalg.Vector(stateObj.(VecStateGrad))
			downstream[i*cells : (i+1)*cells].Add(state)
		}
	}
	for i, uVec := range u {
		downstream[i*cells : (i+1)*cells].Add(uVec)
	}
	for _, v := range g.InStates {
		grad[v] = make(linalg.Vector, len(v.Vector))
	}

	g.Output.PropagateGradient(downstream, grad)

	stateDown := make([]StateGrad, len(g.InStates))
	for i, v := range g.InStates {
		stateDown[i] = VecStateGrad(grad[v])
		delete(grad, v)
	}
	return stateDown
}

type gruRResult struct {
	InStates []*autofunc.Variable
	Output   autofunc.RResult
}

func (g *gruRResult) Outputs() []linalg.Vector {
	return splitVectors(g.Output.Output(), len(g.InStates))
}

func (g *gruRResult) ROutputs() []linalg.Vector {
	return splitVectors(g.Output.ROutput(), len(g.InStates))
}

func (g *gruRResult) RStates() []RState {
	var res []RState
	outsR := g.ROutputs()
	for i, stateVec := range g.Outputs() {
		res = append(res, VecRState{State: stateVec, RState: outsR[i]})
	}
	return res
}

func (g *gruRResult) PropagateRGradient(u, uR []linalg.Vector, s []RStateGrad,
	rg autofunc.RGradient, grad autofunc.Gradient) []RStateGrad {
	if len(g.InStates) == 0 {
		return nil
	}
	if grad == nil {
		grad = autofunc.Gradient{}
	}
	downstream := make(linalg.Vector, len(g.Output.Output()))
	downstreamR := make(linalg.Vector, len(g.Output.Output()))
	cells := len(downstream) / len(g.InStates)
	for i, stateObj := range s {
		if stateObj != nil {
			state := stateObj.(VecRStateGrad)
			downstream[i*cells : (i+1)*cells].Add(state.State)
			downstreamR[i*cells : (i+1)*cells].Add(state.RState)
		}
	}
	for i, uVec := range u {
		downstream[i*cells : (i+1)*cells].Add(uVec)
		downstreamR[i*cells : (i+1)*cells].Add(uR[i])
	}
	for _, v := range g.InStates {
		grad[v] = make(linalg.Vector, len(v.Vector))
		rg[v] = make(linalg.Vector, len(v.Vector))
	}

	g.Output.PropagateRGradient(downstream, downstreamR, rg, grad)

	stateDown := make([]RStateGrad, len(g.InStates))
	for i, v := range g.InStates {
		stateDown[i] = VecRStateGrad{State: grad[v], RState: rg[v]}
		delete(grad, v)
		delete(rg, v)
	}
	return stateDown
}

func splitVectors(in linalg.Vector, n int) []linalg.Vector {
	res := autofunc.Split(n, &autofunc.Variable{Vector: in})
	resList := make([]linalg.Vector, len(res))
	for i, x := range res {
		resList[i] = x.Output()
	}
	return resList
}
