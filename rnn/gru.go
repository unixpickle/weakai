package rnn

import (
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

// GRU is a Block that implements an GRU unit, as
// defined in http://arxiv.org/pdf/1406.1078v3.pdf.
type GRU struct {
	hiddenSize int
	inputValue *lstmGate
	resetGate  *lstmGate
	updateGate *lstmGate
}

// NewGRU creates a GRU with randomly initialized
// weights and biases.
func NewGRU(inputSize, hiddenSize int) *GRU {
	res := &GRU{
		hiddenSize: hiddenSize,
		inputValue: newLSTMGate(inputSize, hiddenSize, &neuralnet.HyperbolicTangent{}),
		resetGate:  newLSTMGate(inputSize, hiddenSize, &neuralnet.Sigmoid{}),
		updateGate: newLSTMGate(inputSize, hiddenSize, &neuralnet.Sigmoid{}),
	}
	return res
}

// DeserializeGRU creates a GRU from some serialized
// data about the GRU.
func DeserializeGRU(d []byte) (serializer.Serializer, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 4 {
		return nil, errors.New("invalid slice length in GRU")
	}
	hiddenSize, ok := slice[0].(serializer.Int)
	inputValue, ok1 := slice[1].(*lstmGate)
	resetGate, ok2 := slice[2].(*lstmGate)
	updateGate, ok3 := slice[3].(*lstmGate)
	if !ok || !ok1 || !ok2 || !ok3 {
		return nil, errors.New("invalid types in GRU slice")
	}
	return &GRU{
		hiddenSize: int(hiddenSize),
		inputValue: inputValue,
		resetGate:  resetGate,
		updateGate: updateGate,
	}, nil
}

// Parameters returns the GRU's parameters in the
// following order: input weights, input biases,
// reset gate weights, reset gate biases, update
// gate weights, update gate biases.
func (g *GRU) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{
		g.inputValue.Dense.Weights.Data,
		g.inputValue.Dense.Biases.Var,
		g.resetGate.Dense.Weights.Data,
		g.resetGate.Dense.Biases.Var,
		g.updateGate.Dense.Weights.Data,
		g.updateGate.Dense.Biases.Var,
	}
}

func (g *GRU) StateSize() int {
	return g.hiddenSize
}

func (g *GRU) Batch(in *BlockInput) BlockOutput {
	n := len(in.Inputs)

	gateInput := joinBlockInput(in)
	stateIn := joinVariables(in.States)

	resetMask := g.resetGate.Batch(gateInput, n)
	updateMask := g.updateGate.Batch(gateInput, n)

	maskedByReset := autofunc.Mul(resetMask, stateIn)
	inputValue := autofunc.Pool(maskedByReset, func(m autofunc.Result) autofunc.Result {
		return g.inputValue.Batch(joinGRUMaskedStates(in.Inputs, m), n)
	})

	newState := autofunc.Pool(updateMask, func(umask autofunc.Result) autofunc.Result {
		updateComplement := autofunc.AddScaler(autofunc.Scale(umask, -1), 1)
		return autofunc.Add(autofunc.Mul(umask, stateIn),
			autofunc.Mul(updateComplement, inputValue))
	})

	return &gruOutput{
		LaneCount: n,
		Output:    newState,
	}
}

func (g *GRU) BatchR(v autofunc.RVector, in *BlockRInput) BlockROutput {
	n := len(in.Inputs)

	gateInput := joinBlockRInput(in)
	stateIn := joinRVariables(in.States)

	resetMask := g.resetGate.BatchR(v, gateInput, n)
	updateMask := g.updateGate.BatchR(v, gateInput, n)

	maskedByReset := autofunc.MulR(resetMask, stateIn)
	inputValue := autofunc.PoolR(maskedByReset, func(m autofunc.RResult) autofunc.RResult {
		return g.inputValue.BatchR(v, joinGRUMaskedRStates(in.Inputs, m), n)
	})

	newState := autofunc.PoolR(updateMask, func(umask autofunc.RResult) autofunc.RResult {
		updateComplement := autofunc.AddScalerR(autofunc.ScaleR(umask, -1), 1)
		return autofunc.AddR(autofunc.MulR(umask, stateIn),
			autofunc.MulR(updateComplement, inputValue))
	})

	return &gruROutput{
		LaneCount: n,
		Output:    newState,
	}
}

func (g *GRU) Serialize() ([]byte, error) {
	slist := []serializer.Serializer{
		serializer.Int(g.hiddenSize),
		g.inputValue,
		g.resetGate,
		g.updateGate,
	}
	return serializer.SerializeSlice(slist)
}

func (g *GRU) SerializerType() string {
	return serializerTypeGRU
}

type gruOutput struct {
	LaneCount int
	Output    autofunc.Result
}

func (g *gruOutput) Outputs() []linalg.Vector {
	return splitVectors(g.Output.Output(), g.LaneCount)
}

func (g *gruOutput) States() []linalg.Vector {
	return splitVectors(g.Output.Output(), g.LaneCount)
}

func (g *gruOutput) Gradient(u *UpstreamGradient, grad autofunc.Gradient) {
	var upstreamGrad linalg.Vector
	if u.States != nil {
		upstreamGrad = joinVectors(u.States)
		if u.Outputs != nil {
			upstreamGrad.Add(joinVectors(u.Outputs))
		}
	} else {
		upstreamGrad = joinVectors(u.Outputs)
	}
	g.Output.PropagateGradient(upstreamGrad, grad)
}

type gruROutput struct {
	LaneCount int
	Output    autofunc.RResult
}

func (g *gruROutput) Outputs() []linalg.Vector {
	return splitVectors(g.Output.Output(), g.LaneCount)
}

func (g *gruROutput) ROutputs() []linalg.Vector {
	return splitVectors(g.Output.ROutput(), g.LaneCount)
}

func (g *gruROutput) States() []linalg.Vector {
	return splitVectors(g.Output.Output(), g.LaneCount)
}

func (g *gruROutput) RStates() []linalg.Vector {
	return splitVectors(g.Output.ROutput(), g.LaneCount)
}

func (g *gruROutput) RGradient(u *UpstreamRGradient, rg autofunc.RGradient,
	grad autofunc.Gradient) {
	var upstreamGrad, upstreamRGrad linalg.Vector
	if u.States != nil {
		upstreamGrad = joinVectors(u.States)
		upstreamRGrad = joinVectors(u.RStates)
		if u.Outputs != nil {
			upstreamGrad.Add(joinVectors(u.Outputs))
			upstreamRGrad.Add(joinVectors(u.ROutputs))
		}
	} else {
		upstreamGrad = joinVectors(u.Outputs)
		upstreamRGrad = joinVectors(u.ROutputs)
	}
	g.Output.PropagateRGradient(upstreamGrad, upstreamRGrad, rg, grad)
}

func joinGRUMaskedStates(inputs []*autofunc.Variable,
	maskedState autofunc.Result) autofunc.Result {
	stateSize := len(maskedState.Output()) / len(inputs)
	var parts []autofunc.Result
	for i, v := range inputs {
		parts = append(parts, v, autofunc.Slice(maskedState, i*stateSize,
			(i+1)*stateSize))
	}
	return autofunc.Concat(parts...)
}

func joinGRUMaskedRStates(inputs []*autofunc.RVariable,
	maskedState autofunc.RResult) autofunc.RResult {
	stateSize := len(maskedState.Output()) / len(inputs)
	var parts []autofunc.RResult
	for i, v := range inputs {
		parts = append(parts, v, autofunc.SliceR(maskedState, i*stateSize,
			(i+1)*stateSize))
	}
	return autofunc.ConcatR(parts...)
}
