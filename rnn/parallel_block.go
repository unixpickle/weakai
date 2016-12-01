package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

func init() {
	var p ParallelBlock
	serializer.RegisterTypedDeserializer(p.SerializerType(), DeserializeParallelBlock)
}

// A ParallelBlock runs two or more blocks in parallel,
// joining their results back together at the end.
//
// Each block is fed the entire input vector, but it is
// trivial to make each block look at a different part of
// the input (e.g. by using a StackedBlock).
type ParallelBlock []Block

// DeserializeParallelBlock deserializes a ParallelBlock.
func DeserializeParallelBlock(d []byte) (ParallelBlock, error) {
	sb, err := DeserializeStackedBlock(d)
	if err != nil {
		return nil, err
	}
	return ParallelBlock(sb), nil
}

// StartState returns an aggregate start state.
func (p ParallelBlock) StartState() State {
	p.assertNotEmpty()
	return StackedBlock(p).StartState()
}

// StartRState returns an aggregate start state.
func (p ParallelBlock) StartRState(rv autofunc.RVector) RState {
	p.assertNotEmpty()
	return StackedBlock(p).StartRState(rv)
}

// PropagateStart propagates through the start state.
func (p ParallelBlock) PropagateStart(s []State, u []StateGrad, g autofunc.Gradient) {
	p.assertNotEmpty()
	StackedBlock(p).PropagateStart(s, u, g)
}

// PropagateStartR propagates through the start state.
func (p ParallelBlock) PropagateStartR(s []RState, u []RStateGrad, rg autofunc.RGradient,
	g autofunc.Gradient) {
	p.assertNotEmpty()
	StackedBlock(p).PropagateStartR(s, u, rg, g)
}

// ApplyBlock applies the sub-blocks and joins their
// results together.
func (p ParallelBlock) ApplyBlock(s []State, in []autofunc.Result) BlockResult {
	inPool := make([]*autofunc.Variable, len(in))
	poolRes := make([]autofunc.Result, len(in))
	for i, x := range in {
		inPool[i] = &autofunc.Variable{Vector: x.Output()}
		poolRes[i] = inPool[i]
	}

	joinedStates := make([][]State, len(in))
	joinedOut := make([]linalg.Vector, len(in))
	subResults := make([]BlockResult, len(p))
	for blockIdx, block := range p {
		subStates := make([]State, len(s))
		for lane, x := range s {
			subStates[lane] = x.([]State)[blockIdx]
		}
		res := block.ApplyBlock(subStates, poolRes)
		subResults[blockIdx] = res
		for lane, outS := range res.States() {
			joinedStates[lane] = append(joinedStates[lane], outS)
		}
		for lane, outV := range res.Outputs() {
			joinedOut[lane] = append(joinedOut[lane], outV...)
		}
	}

	states := make([]State, len(joinedStates))
	for i, x := range joinedStates {
		states[i] = x
	}

	return &parallelBlockResult{
		Inputs:       in,
		InPool:       inPool,
		BlockOuts:    subResults,
		JoinedOuts:   joinedOut,
		JoinedStates: states,
	}
}

// ApplyBlockR applies the sub-blocks and joins their
// results together.
func (p ParallelBlock) ApplyBlockR(rv autofunc.RVector, s []RState,
	in []autofunc.RResult) BlockRResult {
	inPool := make([]*autofunc.Variable, len(in))
	poolRes := make([]autofunc.RResult, len(in))
	for i, x := range in {
		inPool[i] = &autofunc.Variable{Vector: x.Output()}
		poolRes[i] = &autofunc.RVariable{
			Variable:   inPool[i],
			ROutputVec: x.ROutput(),
		}
	}

	joinedStates := make([][]RState, len(in))
	joinedOut := make([]linalg.Vector, len(in))
	joinedOutR := make([]linalg.Vector, len(in))
	subResults := make([]BlockRResult, len(p))
	for blockIdx, block := range p {
		subStates := make([]RState, len(s))
		for lane, x := range s {
			subStates[lane] = x.([]RState)[blockIdx]
		}
		res := block.ApplyBlockR(rv, subStates, poolRes)
		subResults[blockIdx] = res
		for lane, outS := range res.RStates() {
			joinedStates[lane] = append(joinedStates[lane], outS)
		}
		for lane, outV := range res.Outputs() {
			joinedOut[lane] = append(joinedOut[lane], outV...)
		}
		for lane, outV := range res.ROutputs() {
			joinedOutR[lane] = append(joinedOutR[lane], outV...)
		}
	}

	states := make([]RState, len(joinedStates))
	for i, x := range joinedStates {
		states[i] = x
	}

	return &parallelBlockRResult{
		Inputs:       in,
		InPool:       inPool,
		BlockOuts:    subResults,
		JoinedOuts:   joinedOut,
		JoinedOutsR:  joinedOutR,
		JoinedStates: states,
	}
}

// Parameters returns all the parameters in all of the
// sub-blocks which are sgd.Learners.
func (p ParallelBlock) Parameters() []*autofunc.Variable {
	return StackedBlock(p).Parameters()
}

// SerializerType returns the unique ID used to serialize
// a ParallelBlock with the serializer package.
func (p ParallelBlock) SerializerType() string {
	return "github.com/unixpickle/weakai/rnn.ParallelBlock"
}

// Serialize serializes the block.
func (p ParallelBlock) Serialize() ([]byte, error) {
	return StackedBlock(p).Serialize()
}

func (p ParallelBlock) assertNotEmpty() {
	if len(p) == 0 {
		panic("cannot use an empty ParallelBlock")
	}
}

type parallelBlockResult struct {
	Inputs    []autofunc.Result
	InPool    []*autofunc.Variable
	BlockOuts []BlockResult

	JoinedOuts   []linalg.Vector
	JoinedStates []State
}

func (p *parallelBlockResult) Outputs() []linalg.Vector {
	return p.JoinedOuts
}

func (p *parallelBlockResult) States() []State {
	return p.JoinedStates
}

func (p *parallelBlockResult) PropagateGradient(u []linalg.Vector, s []StateGrad,
	g autofunc.Gradient) []StateGrad {
	for _, v := range p.InPool {
		g[v] = make(linalg.Vector, len(v.Vector))
	}
	n := len(p.JoinedOuts)
	upstreamOffsets := make([]int, n)
	downstreams := make([][]StateGrad, n)
	for blockIdx, blockOut := range p.BlockOuts {
		outs := blockOut.Outputs()
		var subUpstream []linalg.Vector
		if u != nil {
			subUpstream = make([]linalg.Vector, len(u))
			for lane, uVec := range u {
				outLen := len(outs[lane])
				startIdx := upstreamOffsets[lane]
				subUpstream[lane] = uVec[startIdx : startIdx+outLen]
				upstreamOffsets[lane] += outLen
			}
		}
		var subState []StateGrad
		if s != nil {
			subState = make([]StateGrad, len(s))
			for lane, upState := range s {
				if upState != nil {
					subState[lane] = upState.([]StateGrad)[blockIdx]
				}
			}
		}
		ds := blockOut.PropagateGradient(subUpstream, subState, g)
		for lane, x := range ds {
			downstreams[lane] = append(downstreams[lane], x)
		}
	}
	for i, v := range p.InPool {
		vec := g[v]
		delete(g, v)
		p.Inputs[i].PropagateGradient(vec, g)
	}
	var res []StateGrad
	for _, x := range downstreams {
		res = append(res, x)
	}
	return res
}

type parallelBlockRResult struct {
	Inputs    []autofunc.RResult
	InPool    []*autofunc.Variable
	BlockOuts []BlockRResult

	JoinedOuts   []linalg.Vector
	JoinedOutsR  []linalg.Vector
	JoinedStates []RState
}

func (p *parallelBlockRResult) Outputs() []linalg.Vector {
	return p.JoinedOuts
}

func (p *parallelBlockRResult) ROutputs() []linalg.Vector {
	return p.JoinedOutsR
}

func (p *parallelBlockRResult) RStates() []RState {
	return p.JoinedStates
}

func (p *parallelBlockRResult) PropagateRGradient(u, uR []linalg.Vector, s []RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) []RStateGrad {
	if g == nil {
		g = autofunc.Gradient{}
	}
	for _, v := range p.InPool {
		g[v] = make(linalg.Vector, len(v.Vector))
		rg[v] = make(linalg.Vector, len(v.Vector))
	}
	n := len(p.JoinedOuts)
	upstreamOffsets := make([]int, n)
	downstreams := make([][]RStateGrad, n)
	for blockIdx, blockOut := range p.BlockOuts {
		outs := blockOut.Outputs()
		var subUpstream, subUpstreamR []linalg.Vector
		if u != nil {
			subUpstream = make([]linalg.Vector, len(u))
			subUpstreamR = make([]linalg.Vector, len(u))
			for lane, uVec := range u {
				outLen := len(outs[lane])
				startIdx := upstreamOffsets[lane]
				subUpstream[lane] = uVec[startIdx : startIdx+outLen]
				subUpstreamR[lane] = uR[lane][startIdx : startIdx+outLen]
				upstreamOffsets[lane] += outLen
			}
		}
		var subState []RStateGrad
		if s != nil {
			subState = make([]RStateGrad, len(s))
			for lane, upState := range s {
				if upState != nil {
					subState[lane] = upState.([]RStateGrad)[blockIdx]
				}
			}
		}
		ds := blockOut.PropagateRGradient(subUpstream, subUpstreamR, subState, rg, g)
		for lane, x := range ds {
			downstreams[lane] = append(downstreams[lane], x)
		}
	}
	for i, v := range p.InPool {
		vec := g[v]
		vecR := rg[v]
		delete(g, v)
		delete(rg, v)
		p.Inputs[i].PropagateRGradient(vec, vecR, rg, g)
	}
	var res []RStateGrad
	for _, x := range downstreams {
		res = append(res, x)
	}
	return res
}
