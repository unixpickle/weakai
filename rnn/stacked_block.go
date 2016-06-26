package rnn

import (
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
)

// A StackedBlock is a Block which feeds the output
// from each sub-block into the input of the next
// sub-block, essentially "stacking" blocks.
// It can be used to form deep RNNs.
type StackedBlock []Block

func DeserializeStackedBlock(d []byte) (serializer.Serializer, error) {
	list, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	res := make(StackedBlock, len(list))
	for i, s := range list {
		var ok bool
		res[i], ok = s.(Block)
		if !ok {
			return nil, fmt.Errorf("layer %d (%T) is not Block", i, s)
		}
	}
	return res, nil
}

// StateSize returns the sum of the state sizes of
// all the sub-blocks, since states from a
// StackedBlock are just concatenations of the
// states of the sub-blocks.
func (d StackedBlock) StateSize() int {
	var res int
	for _, x := range d {
		res += x.StateSize()
	}
	return res
}

func (d StackedBlock) Batch(in *BlockInput) BlockOutput {
	states := d.subBlockStates(varsToVecs(in.States))
	result := &stackedBlockOutput{
		StackedBlock: d,
		InStates:     in.States,
		Inputs:       in.Inputs,
		StatesVec:    make([]linalg.Vector, len(in.Inputs)),
	}
	for subBlockIdx, subBlock := range d {
		input := &BlockInput{States: vecsToVars(states[subBlockIdx])}
		if subBlockIdx == 0 {
			input.Inputs = in.Inputs
		} else {
			input.Inputs = vecsToVars(result.BlockOutputs[subBlockIdx-1].Outputs())
		}
		output := subBlock.Batch(input)
		result.BlockInputs = append(result.BlockInputs, input)
		result.BlockOutputs = append(result.BlockOutputs, output)
		for i, x := range output.States() {
			result.StatesVec[i] = append(result.StatesVec[i], x...)
		}
	}
	return result
}

func (d StackedBlock) BatchR(context autofunc.RVector, in *BlockRInput) BlockROutput {
	inStates, inRStates := rvarsToVecs(in.States)
	states := d.subBlockStates(inStates)
	rstates := d.subBlockStates(inRStates)
	result := &stackedBlockROutput{
		StackedBlock: d,
		InStates:     in.States,
		Inputs:       in.Inputs,
		StatesVec:    make([]linalg.Vector, len(in.Inputs)),
		RStatesVec:   make([]linalg.Vector, len(in.Inputs)),
	}
	for subBlockIdx, subBlock := range d {
		input := &BlockRInput{
			States: vecsToRVars(states[subBlockIdx], rstates[subBlockIdx]),
		}
		if subBlockIdx == 0 {
			input.Inputs = in.Inputs
		} else {
			lastOut := result.BlockOutputs[subBlockIdx-1]
			input.Inputs = vecsToRVars(lastOut.Outputs(), lastOut.ROutputs())
		}
		output := subBlock.BatchR(context, input)
		result.BlockInputs = append(result.BlockInputs, input)
		result.BlockOutputs = append(result.BlockOutputs, output)
		for i, x := range output.States() {
			result.StatesVec[i] = append(result.StatesVec[i], x...)
		}
		for i, x := range output.RStates() {
			result.RStatesVec[i] = append(result.RStatesVec[i], x...)
		}
	}
	return result
}

// Parameters returns the parameters every Learner
// sub-block of this block.
func (d StackedBlock) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, b := range d {
		if l, ok := b.(sgd.Learner); ok {
			res = append(res, l.Parameters()...)
		}
	}
	return res
}

// Serialize attempts to serialize all of the sub-blocks
// if they implement the Serializer interface.
func (d StackedBlock) Serialize() ([]byte, error) {
	serializers := make([]serializer.Serializer, len(d))
	for i, l := range d {
		if s, ok := l.(serializer.Serializer); ok {
			serializers[i] = s
		} else {
			return nil, fmt.Errorf("layer %d (%T) is not a Serializer", i, l)
		}
	}
	return serializer.SerializeSlice(serializers)
}

func (d StackedBlock) SerializerType() string {
	return serializerTypeStackedBlock
}

// subBlockStates splits a slice of packed states into
// a slice of state slices, on per sub-block.
func (d StackedBlock) subBlockStates(states []linalg.Vector) [][]linalg.Vector {
	var res [][]linalg.Vector
	var curIdx int
	for _, subBlock := range d {
		stateLen := subBlock.StateSize()
		var batchParts []linalg.Vector
		for _, u := range states {
			out := u[curIdx : curIdx+stateLen]
			batchParts = append(batchParts, out)
		}
		res = append(res, batchParts)
		curIdx += stateLen
	}
	return res
}

type stackedBlockOutput struct {
	StackedBlock StackedBlock

	InStates []*autofunc.Variable
	Inputs   []*autofunc.Variable

	BlockInputs  []*BlockInput
	BlockOutputs []BlockOutput

	StatesVec []linalg.Vector
}

func (d *stackedBlockOutput) States() []linalg.Vector {
	return d.StatesVec
}

func (d *stackedBlockOutput) Outputs() []linalg.Vector {
	return d.BlockOutputs[len(d.BlockOutputs)-1].Outputs()
}

func (d *stackedBlockOutput) Gradient(u *UpstreamGradient, g autofunc.Gradient) {
	if u.Outputs == nil && u.States == nil {
		return
	}

	var upstreamStateGrads [][]linalg.Vector
	if u.States != nil {
		upstreamStateGrads = d.StackedBlock.subBlockStates(u.States)
	}

	d.setupDownstreamStateGrads(g)

	currentUpstream := &UpstreamGradient{Outputs: u.Outputs}
	if upstreamStateGrads != nil {
		currentUpstream.States = upstreamStateGrads[len(upstreamStateGrads)-1]
	}

	for i := len(d.BlockInputs) - 1; i >= 0; i-- {
		o := d.BlockOutputs[i]
		if i != 0 {
			for _, in := range d.BlockInputs[i].Inputs {
				g[in] = make(linalg.Vector, len(in.Vector))
			}
		}
		o.Gradient(currentUpstream, g)
		if i != 0 {
			currentUpstream.Outputs = nil
			for _, in := range d.BlockInputs[i].Inputs {
				currentUpstream.Outputs = append(currentUpstream.Outputs, g[in])
				delete(g, in)
			}
			if upstreamStateGrads != nil {
				currentUpstream.States = upstreamStateGrads[i-1]
			}
		}
	}

	d.joinDownstreamStateGrads(g)
}

func (d *stackedBlockOutput) setupDownstreamStateGrads(g autofunc.Gradient) {
	for lane, inState := range d.InStates {
		if _, ok := g[inState]; ok {
			for _, in := range d.BlockInputs {
				s := in.States[lane]
				g[s] = make(linalg.Vector, len(s.Vector))
			}
		}
	}
}

func (d *stackedBlockOutput) joinDownstreamStateGrads(g autofunc.Gradient) {
	for lane, inState := range d.InStates {
		if stateGrad, ok := g[inState]; ok {
			stateIdx := 0
			for _, in := range d.BlockInputs {
				s := in.States[lane]
				substateGrad := g[s]
				stateGrad[stateIdx : stateIdx+len(substateGrad)].Add(substateGrad)
				delete(g, s)
				stateIdx += len(substateGrad)
			}
		}
	}
}

type stackedBlockROutput struct {
	StackedBlock StackedBlock

	InStates []*autofunc.RVariable
	Inputs   []*autofunc.RVariable

	BlockInputs  []*BlockRInput
	BlockOutputs []BlockROutput

	StatesVec  []linalg.Vector
	RStatesVec []linalg.Vector
}

func (d *stackedBlockROutput) States() []linalg.Vector {
	return d.StatesVec
}

func (d *stackedBlockROutput) RStates() []linalg.Vector {
	return d.RStatesVec
}

func (d *stackedBlockROutput) Outputs() []linalg.Vector {
	return d.BlockOutputs[len(d.BlockOutputs)-1].Outputs()
}

func (d *stackedBlockROutput) ROutputs() []linalg.Vector {
	return d.BlockOutputs[len(d.BlockOutputs)-1].ROutputs()
}

func (d *stackedBlockROutput) RGradient(u *UpstreamRGradient, rg autofunc.RGradient,
	g autofunc.Gradient) {
	// Needed so we can store temporary values in g.
	if g == nil {
		g = autofunc.Gradient{}
	}

	var upstreamStateGrads [][]linalg.Vector
	var upstreamStateRGrads [][]linalg.Vector
	if u.States != nil {
		upstreamStateGrads = d.StackedBlock.subBlockStates(u.States)
		upstreamStateRGrads = d.StackedBlock.subBlockStates(u.RStates)
	}

	d.setupDownstreamStateGrads(g)
	d.setupDownstreamStateGrads(rg)

	currentUpstream := &UpstreamRGradient{
		UpstreamGradient: UpstreamGradient{Outputs: u.Outputs},
		ROutputs:         u.ROutputs,
	}
	if upstreamStateGrads != nil {
		currentUpstream.States = upstreamStateGrads[len(upstreamStateGrads)-1]
		currentUpstream.RStates = upstreamStateRGrads[len(upstreamStateRGrads)-1]
	}

	for i := len(d.BlockInputs) - 1; i >= 0; i-- {
		o := d.BlockOutputs[i]
		if i != 0 {
			for _, in := range d.BlockInputs[i].Inputs {
				g[in.Variable] = make(linalg.Vector, len(in.Variable.Vector))
				rg[in.Variable] = make(linalg.Vector, len(in.Variable.Vector))
			}
		}
		o.RGradient(currentUpstream, rg, g)
		if i != 0 {
			currentUpstream.Outputs = nil
			currentUpstream.ROutputs = nil
			for _, in := range d.BlockInputs[i].Inputs {
				currentUpstream.Outputs = append(currentUpstream.Outputs, g[in.Variable])
				currentUpstream.ROutputs = append(currentUpstream.ROutputs, rg[in.Variable])
				delete(g, in.Variable)
				delete(rg, in.Variable)
			}
			if upstreamStateGrads != nil {
				currentUpstream.States = upstreamStateGrads[i-1]
				currentUpstream.RStates = upstreamStateRGrads[i-1]
			}
		}
	}

	d.joinDownstreamStateGrads(g)
	d.joinDownstreamStateGrads(rg)
}

func (d *stackedBlockROutput) setupDownstreamStateGrads(g map[*autofunc.Variable]linalg.Vector) {
	for lane, inState := range d.InStates {
		if _, ok := g[inState.Variable]; ok {
			for _, in := range d.BlockInputs {
				s := in.States[lane].Variable
				g[s] = make(linalg.Vector, len(s.Vector))
			}
		}
	}
}

func (d *stackedBlockROutput) joinDownstreamStateGrads(g map[*autofunc.Variable]linalg.Vector) {
	for lane, inState := range d.InStates {
		if stateGrad, ok := g[inState.Variable]; ok {
			stateIdx := 0
			for _, in := range d.BlockInputs {
				s := in.States[lane].Variable
				substateGrad := g[s]
				stateGrad[stateIdx : stateIdx+len(substateGrad)].Add(substateGrad)
				delete(g, s)
				stateIdx += len(substateGrad)
			}
		}
	}
}

func varsToVecs(vs []*autofunc.Variable) []linalg.Vector {
	res := make([]linalg.Vector, len(vs))
	for i, x := range vs {
		res[i] = x.Vector
	}
	return res
}

func rvarsToVecs(vs []*autofunc.RVariable) (vectors, rvectors []linalg.Vector) {
	vectors = make([]linalg.Vector, len(vs))
	rvectors = make([]linalg.Vector, len(vs))
	for i, x := range vs {
		vectors[i] = x.Output()
		rvectors[i] = x.ROutput()
	}
	return
}

func vecsToVars(vs []linalg.Vector) []*autofunc.Variable {
	res := make([]*autofunc.Variable, len(vs))
	for i, x := range vs {
		res[i] = &autofunc.Variable{Vector: x}
	}
	return res
}

func vecsToRVars(vecs, rvecs []linalg.Vector) []*autofunc.RVariable {
	res := make([]*autofunc.RVariable, len(vecs))
	for i, x := range vecs {
		v := &autofunc.Variable{Vector: x}
		res[i] = &autofunc.RVariable{Variable: v, ROutputVec: rvecs[i]}
	}
	return res
}
