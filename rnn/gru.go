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
func (g *GRU) PropagateStart(_ []State, s []StateGrad, grad autofunc.Gradient) {
	PropagateVarState(g.initState, s, grad)
}

// PropagateStartR propagates through the start state.
func (g *GRU) PropagateStartR(_ []RState, s []RStateGrad, rg autofunc.RGradient,
	grad autofunc.Gradient) {
	PropagateVarStateR(g.initState, s, rg, grad)
}

// ApplyBlock applies the block to an input.
func (g *GRU) ApplyBlock(s []State, in []autofunc.Result) BlockResult {
	stateVars, stateRes := PoolVecStates(s)
	var gateInputs []autofunc.Result
	for i, x := range stateRes {
		gateInputs = append(gateInputs, in[i], x)
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
	stateVars, stateRes := PoolVecRStates(s)
	var gateInputs []autofunc.RResult
	for i, x := range stateRes {
		gateInputs = append(gateInputs, in[i], x)
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
	return PropagateVecStatePool(grad, g.InStates, func() {
		g.Output.PropagateGradient(downstream, grad)
	})
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
	return PropagateVecRStatePool(rg, grad, g.InStates, func() {
		g.Output.PropagateRGradient(downstream, downstreamR, rg, grad)
	})
}

func splitVectors(in linalg.Vector, n int) []linalg.Vector {
	res := autofunc.Split(n, &autofunc.Variable{Vector: in})
	resList := make([]linalg.Vector, len(res))
	for i, x := range res {
		resList[i] = x.Output()
	}
	return resList
}
