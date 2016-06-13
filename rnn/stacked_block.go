package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A StackedBlock is a Block which feeds the output
// from each sub-blocks into the input of the next
// sub-block, essentially "stacking" blocks.
// It can be used to form deep RNNs.
type StackedBlock []Block

func (d StackedBlock) StateSize() int {
	var res int
	for _, x := range d {
		res += x.StateSize()
	}
	return res
}

func (d StackedBlock) Batch(in *BlockInput) BlockOutput {
	states := d.splitStates(varsToVecs(in.States))
	result := &stackedBlockOutput{
		StackedBlock: d,
		InStates:     in.States,
		Inputs:       in.Inputs,
	}
	for i, b := range d {
		input := &BlockInput{States: vecsToVars(states[i])}
		if i == 0 {
			input.Inputs = in.Inputs
		} else {
			for _, x := range result.BlockOutputs[i-1].Outputs() {
				v := &autofunc.Variable{Vector: x}
				input.Inputs = append(input.Inputs, v)
			}
		}
		output := b.Batch(input)
		result.BlockInputs = append(result.BlockInputs, input)
		result.BlockOutputs = append(result.BlockOutputs, output)
		result.StatesVec = append(result.StatesVec, output.States()...)
	}
	return result
}

func (d StackedBlock) BatchR(context autofunc.RVector, in *BlockRInput) BlockROutput {
	inStates, inRStates := rvarsToVecs(in.States)
	states := d.splitStates(inStates)
	rstates := d.splitStates(inRStates)
	result := &stackedBlockROutput{
		StackedBlock: d,
		InStates:     in.States,
		Inputs:       in.Inputs,
	}
	for i, b := range d {
		input := &BlockRInput{States: vecsToRVars(states[i], rstates[i])}
		if i == 0 {
			input.Inputs = in.Inputs
		} else {
			rOuts := result.BlockOutputs[i-1].ROutputs()
			for i, x := range result.BlockOutputs[i-1].Outputs() {
				v := &autofunc.Variable{Vector: x}
				rv := &autofunc.RVariable{Variable: v, ROutputVec: rOuts[i]}
				input.Inputs = append(input.Inputs, rv)
			}
		}
		output := b.BatchR(context, input)
		result.BlockInputs = append(result.BlockInputs, input)
		result.BlockOutputs = append(result.BlockOutputs, output)
		result.StatesVec = append(result.StatesVec, output.States()...)
		result.RStatesVec = append(result.RStatesVec, output.RStates()...)
	}
	return result
}

// Parameters returns the parameters every Learner
// sub-block of this block.
func (d StackedBlock) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, b := range d {
		if l, ok := b.(Learner); ok {
			res = append(res, l.Parameters()...)
		}
	}
	return res
}

// splitStates splits a slice of states (one per lane)
// into a slice of slices of states, where each inner
// slice corresponds to one block, and each vector in
// an inner slice corresponds to the state of one lane.
func (d StackedBlock) splitStates(states []linalg.Vector) [][]linalg.Vector {
	var res [][]linalg.Vector
	var curIdx int
	for _, b := range d {
		stateLen := b.StateSize()
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

	var stateGrads [][]linalg.Vector
	if u.States != nil {
		stateGrads = d.StackedBlock.splitStates(u.States)
	}

	for lane, inState := range d.InStates {
		if _, ok := g[inState]; ok {
			for _, in := range d.BlockInputs {
				s := in.States[lane]
				g[s] = make(linalg.Vector, len(s.Vector))
			}
		}
	}

	currentUpstream := &UpstreamGradient{Outputs: u.Outputs}
	if stateGrads != nil {
		currentUpstream.States = stateGrads[len(stateGrads)-1]
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
			for i, in := range d.BlockInputs[i].Inputs {
				currentUpstream.Outputs[i] = g[in]
				delete(g, in)
			}
			if stateGrads != nil {
				currentUpstream.States = stateGrads[i-1]
			}
		}
	}

	for lane, inState := range d.InStates {
		if stateGrad, ok := g[inState]; ok {
			stateIdx := 0
			for _, in := range d.BlockInputs {
				s := in.States[lane]
				substateGrad := g[s]
				stateGrad[stateIdx : stateIdx+len(substateGrad)].Add(substateGrad)
				delete(g, s)
			}
			stateIdx += len(d.BlockInputs[0].States[lane].Vector)
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

	var stateGrads [][]linalg.Vector
	var stateRGrads [][]linalg.Vector
	if u.States != nil {
		stateGrads = d.StackedBlock.splitStates(u.States)
		stateRGrads = d.StackedBlock.splitStates(u.RStates)
	}

	for lane, inState := range d.InStates {
		if _, ok := g[inState.Variable]; ok {
			for _, in := range d.BlockInputs {
				s := in.States[lane].Variable
				g[s] = make(linalg.Vector, len(s.Vector))
			}
		}
		if _, ok := rg[inState.Variable]; ok {
			for _, in := range d.BlockInputs {
				s := in.States[lane].Variable
				rg[s] = make(linalg.Vector, len(s.Vector))
			}
		}
	}

	currentUpstream := &UpstreamRGradient{
		UpstreamGradient: UpstreamGradient{Outputs: u.Outputs},
		ROutputs:         u.ROutputs,
	}
	if stateGrads != nil {
		currentUpstream.States = stateGrads[len(stateGrads)-1]
		currentUpstream.RStates = stateRGrads[len(stateRGrads)-1]
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
			for i, in := range d.BlockInputs[i].Inputs {
				currentUpstream.Outputs[i] = g[in.Variable]
				currentUpstream.ROutputs[i] = rg[in.Variable]
				delete(g, in.Variable)
				delete(rg, in.Variable)
			}
			if stateGrads != nil {
				currentUpstream.States = stateGrads[i-1]
				currentUpstream.RStates = stateRGrads[i-1]
			}
		}
	}

	d.joinStateGrads(g)
	d.joinStateGrads(rg)
}

func (d *stackedBlockROutput) joinStateGrads(g map[*autofunc.Variable]linalg.Vector) {
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
