package rnn

import (
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
)

func init() {
	var s StackedBlock
	serializer.RegisterTypedDeserializer(s.SerializerType(), DeserializeStackedBlock)
}

// A StackedBlock implements a deep Block which works by
// feeding the output of each block into the input of the
// next block in the stack.
// It is essential for building deep RNNs.
type StackedBlock []Block

// DeserializeStackedBlock deserializes a StackedBlock.
func DeserializeStackedBlock(d []byte) (StackedBlock, error) {
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

// StartState generates a start state which encapsulates
// the start states of all the nested blocks.
func (s StackedBlock) StartState() State {
	if len(s) == 0 {
		panic("cannot use an empty StackedBlock")
	}
	var res []State
	for _, x := range s {
		res = append(res, x.StartState())
	}
	return res
}

// StartRState is like StartState.
func (s StackedBlock) StartRState(rv autofunc.RVector) RState {
	if len(s) == 0 {
		panic("cannot use an empty StackedBlock")
	}
	var res []RState
	for _, b := range s {
		res = append(res, b.StartRState(rv))
	}
	return res
}

// PropagateStart back-propagates through all the child
// Blocks.
func (s StackedBlock) PropagateStart(upstream []StateGrad, g autofunc.Gradient) {
	if len(s) == 0 {
		panic("cannot use an empty StackedBlock")
	}
	for childIdx, child := range s {
		var grad []StateGrad
		for _, seqGrad := range upstream {
			grad = append(grad, seqGrad.([]StateGrad)[childIdx])
		}
		child.PropagateStart(grad, g)
	}
}

// PropagateStartR is like PropagateStart.
func (s StackedBlock) PropagateStartR(upstream []RStateGrad, rg autofunc.RGradient,
	g autofunc.Gradient) {
	if len(s) == 0 {
		panic("cannot use an empty StackedBlock")
	}
	for childIdx, child := range s {
		var grad []RStateGrad
		for _, seqGrad := range upstream {
			grad = append(grad, seqGrad.([]RStateGrad)[childIdx])
		}
		child.PropagateStartR(grad, rg, g)
	}
}

func (s StackedBlock) ApplyBlock(states []State, in []autofunc.Result) BlockResult {
	if len(s) == 0 {
		panic("cannot use an empty StackedBlock")
	}
	res := &stackedBlockResult{
		Depth: len(s),
		Pools: make([][]*autofunc.Variable, len(in)),
	}
	outStates := make([][]State, len(states))
	for i, layer := range s {
		var inState []State
		for _, stateList := range states {
			inState = append(inState, stateList.([]State)[i])
		}
		out := layer.ApplyBlock(inState, in)
		res.Outs = append(res.Outs, out)
		for j, state := range out.States() {
			outStates[j] = append(outStates[j], state)
		}
		if i+1 == len(s) {
			res.OutVecs = out.Outputs()
		} else {
			in = make([]autofunc.Result, len(out.Outputs()))
			for j, outVec := range out.Outputs() {
				poolVar := &autofunc.Variable{Vector: outVec}
				res.Pools[j] = append(res.Pools[j], poolVar)
				in[j] = poolVar
			}
		}
	}
	res.OutStates = make([]State, len(outStates))
	for i, x := range outStates {
		res.OutStates[i] = x
	}
	return res
}

func (s StackedBlock) ApplyBlockR(v autofunc.RVector, states []RState,
	in []autofunc.RResult) BlockRResult {
	if len(s) == 0 {
		panic("cannot use an empty StackedBlock")
	}
	if len(s) == 0 {
		panic("cannot use an empty StackedBlock")
	}
	res := &stackedBlockRResult{
		Depth: len(s),
		Pools: make([][]*autofunc.Variable, len(in)),
	}
	outStates := make([][]RState, len(states))
	for i, layer := range s {
		var inState []RState
		for _, stateList := range states {
			inState = append(inState, stateList.([]RState)[i])
		}
		out := layer.ApplyBlockR(v, inState, in)
		res.Outs = append(res.Outs, out)
		for j, state := range out.RStates() {
			outStates[j] = append(outStates[j], state)
		}
		if i+1 == len(s) {
			res.OutVecs = out.Outputs()
			res.ROutVecs = out.ROutputs()
		} else {
			in = make([]autofunc.RResult, len(out.Outputs()))
			for j, outVec := range out.Outputs() {
				poolVar := &autofunc.Variable{Vector: outVec}
				res.Pools[j] = append(res.Pools[j], poolVar)
				poolRVar := &autofunc.RVariable{
					Variable:   poolVar,
					ROutputVec: out.ROutputs()[j],
				}
				in[j] = poolRVar
			}
		}
	}
	res.OutStates = make([]RState, len(outStates))
	for i, x := range outStates {
		res.OutStates[i] = x
	}
	return res
}

// Parameters returns the parameters of every Learner
// sub-block of this block.
func (s StackedBlock) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, b := range s {
		if l, ok := b.(sgd.Learner); ok {
			res = append(res, l.Parameters()...)
		}
	}
	return res
}

// Serialize attempts to serialize all of the sub-blocks
// if they implement the Serializer interface.
func (s StackedBlock) Serialize() ([]byte, error) {
	serializers := make([]serializer.Serializer, len(s))
	for i, l := range s {
		if ser, ok := l.(serializer.Serializer); ok {
			serializers[i] = ser
		} else {
			return nil, fmt.Errorf("layer %d (%T) is not a Serializer", i, l)
		}
	}
	return serializer.SerializeSlice(serializers)
}

func (s StackedBlock) SerializerType() string {
	return "github.com/unixpickle/weakai/rnn.StackedBlock"
}

type stackedBlockResult struct {
	Depth int
	Outs  []BlockResult

	// One index in each of the following slices corresponds
	// to one of the input vectors.
	Pools     [][]*autofunc.Variable
	OutVecs   []linalg.Vector
	OutStates []State
}

func (s *stackedBlockResult) Outputs() []linalg.Vector {
	return s.OutVecs
}

func (s *stackedBlockResult) States() []State {
	return s.OutStates
}

func (s *stackedBlockResult) PropagateGradient(u []linalg.Vector, su []StateGrad,
	g autofunc.Gradient) []StateGrad {
	stateDownstream := make([][]StateGrad, len(s.Pools))
	for layer := s.Depth - 1; layer >= 0; layer-- {
		var stateUpstream []StateGrad
		for lane, poolVec := range s.Pools {
			if layer > 0 {
				poolVar := poolVec[layer-1]
				g[poolVar] = make(linalg.Vector, len(poolVar.Vector))
			}
			if su != nil {
				if su[lane] != nil {
					laneSU := su[lane].([]StateGrad)
					stateUpstream = append(stateUpstream, laneSU[layer])
				} else {
					stateUpstream = append(stateUpstream, nil)
				}
			}
		}
		downstream := s.Outs[layer].PropagateGradient(u, stateUpstream, g)
		if layer > 0 {
			u = nil
			for _, poolVec := range s.Pools {
				poolVar := poolVec[layer-1]
				u = append(u, g[poolVar])
				delete(g, poolVar)
			}
		}
		for lane, sg := range downstream {
			stateDownstream[lane] = append([]StateGrad{sg}, stateDownstream[lane]...)
		}
	}

	var res []StateGrad
	for _, x := range stateDownstream {
		res = append(res, x)
	}
	return res
}

type stackedBlockRResult struct {
	Depth int
	Outs  []BlockRResult

	// One index in each of the following slices corresponds
	// to one of the input vectors.
	Pools     [][]*autofunc.Variable
	OutVecs   []linalg.Vector
	ROutVecs  []linalg.Vector
	OutStates []RState
}

func (s *stackedBlockRResult) Outputs() []linalg.Vector {
	return s.OutVecs
}

func (s *stackedBlockRResult) ROutputs() []linalg.Vector {
	return s.ROutVecs
}

func (s *stackedBlockRResult) RStates() []RState {
	return s.OutStates
}

func (s *stackedBlockRResult) PropagateRGradient(u, uR []linalg.Vector, su []RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) []RStateGrad {
	if g == nil {
		g = autofunc.Gradient{}
	}
	if (u == nil) != (uR == nil) {
		panic("upstream and upstreamR must match in nil-ness")
	}
	stateDownstream := make([][]RStateGrad, len(s.Pools))
	for layer := s.Depth - 1; layer >= 0; layer-- {
		var stateUpstream []RStateGrad
		for lane, poolVec := range s.Pools {
			if layer > 0 {
				poolVar := poolVec[layer-1]
				g[poolVar] = make(linalg.Vector, len(poolVar.Vector))
				rg[poolVar] = make(linalg.Vector, len(poolVar.Vector))
			}
			if su != nil {
				if su[lane] != nil {
					laneSU := su[lane].([]RStateGrad)
					stateUpstream = append(stateUpstream, laneSU[layer])
				} else {
					stateUpstream = append(stateUpstream, nil)
				}
			}
		}
		downstream := s.Outs[layer].PropagateRGradient(u, uR, stateUpstream, rg, g)
		if layer > 0 {
			u, uR = nil, nil
			for _, poolVec := range s.Pools {
				poolVar := poolVec[layer-1]
				u = append(u, g[poolVar])
				uR = append(uR, rg[poolVar])
				delete(g, poolVar)
				delete(rg, poolVar)
			}
		}
		for lane, sg := range downstream {
			stateDownstream[lane] = append([]RStateGrad{sg}, stateDownstream[lane]...)
		}
	}

	var res []RStateGrad
	for _, x := range stateDownstream {
		res = append(res, x)
	}
	return res
}
