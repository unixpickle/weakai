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

func (d StackedBlock) BatchR(in *BlockInput) BlockOutput {
	// TODO: this.
	return nil
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

func varsToVecs(vs []*autofunc.Variable) []linalg.Vector {
	res := make([]linalg.Vector, len(vs))
	for i, x := range vs {
		res[i] = x.Vector
	}
	return res
}

func vecsToVars(vs []linalg.Vector) []*autofunc.Variable {
	res := make([]*autofunc.Variable, len(vs))
	for i, x := range vs {
		res[i] = &autofunc.Variable{Vector: x}
	}
	return res
}
