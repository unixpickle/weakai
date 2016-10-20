package rnn

import (
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
)

func init() {
	var b BlockSeqFunc
	serializer.RegisterTypedDeserializer(b.SerializerType(), DeserializeBlockSeqFunc)
}

// A BlockSeqFunc creates a seqfunc.RFunc that evaluates
// the Block sequentially.
type BlockSeqFunc struct {
	B Block
}

// DeserializeBlockSeqFunc deserializes a BlockSeqFunc.
func DeserializeBlockSeqFunc(d []byte) (*BlockSeqFunc, error) {
	obj, err := serializer.DeserializeWithType(d)
	if err != nil {
		return nil, err
	}
	block, ok := obj.(Block)
	if !ok {
		return nil, fmt.Errorf("expected Block but got %T", obj)
	}
	return &BlockSeqFunc{B: block}, nil
}

// ApplySeqs evaluates the block on each of the input
// sequences.
func (b *BlockSeqFunc) ApplySeqs(in seqfunc.Result) seqfunc.Result {
	res := &blockSeqFuncResult{
		B:      b.B,
		Start:  make([]State, len(in.OutputSeqs())),
		Input:  in,
		InPool: make([][]*autofunc.Variable, len(in.OutputSeqs())),
		Output: make([][]linalg.Vector, len(in.OutputSeqs())),
	}
	maxLen := maxSeqLength(in.OutputSeqs())
	inStates := map[int]State{}
	for i := range in.OutputSeqs() {
		inStates[i] = b.B.StartState()
		res.Start[i] = inStates[i]
	}
	for t := 0; t < maxLen; t++ {
		var stateIn []State
		var resIn []autofunc.Result
		for lane, seq := range in.OutputSeqs() {
			if len(seq) <= t {
				continue
			}
			inVar := &autofunc.Variable{Vector: seq[t]}
			res.InPool[lane] = append(res.InPool[lane], inVar)
			stateIn = append(stateIn, inStates[lane])
			resIn = append(resIn, inVar)
		}
		out := b.B.ApplyBlock(stateIn, resIn)
		res.StepOuts = append(res.StepOuts, out)
		outVecs := out.Outputs()
		outStates := out.States()
		for lane, seq := range in.OutputSeqs() {
			if len(seq) <= t {
				continue
			}
			res.Output[lane] = append(res.Output[lane], outVecs[0])
			inStates[lane] = outStates[0]
			outVecs = outVecs[1:]
			outStates = outStates[1:]
		}
	}
	return res
}

// ApplySeqsR is like ApplySeqs but for RResults.
func (b *BlockSeqFunc) ApplySeqsR(rv autofunc.RVector, in seqfunc.RResult) seqfunc.RResult {
	res := &blockSeqFuncRResult{
		B:       b.B,
		Input:   in,
		Start:   make([]RState, len(in.OutputSeqs())),
		InPool:  make([][]*autofunc.Variable, len(in.OutputSeqs())),
		Output:  make([][]linalg.Vector, len(in.OutputSeqs())),
		ROutput: make([][]linalg.Vector, len(in.OutputSeqs())),
	}
	maxLen := maxSeqLength(in.OutputSeqs())
	inStates := map[int]RState{}
	for i := range in.OutputSeqs() {
		inStates[i] = b.B.StartRState(rv)
		res.Start[i] = inStates[i]
	}
	for t := 0; t < maxLen; t++ {
		var stateIn []RState
		var resIn []autofunc.RResult
		for lane, seq := range in.OutputSeqs() {
			if len(seq) <= t {
				continue
			}
			inVar := &autofunc.Variable{Vector: seq[t]}
			res.InPool[lane] = append(res.InPool[lane], inVar)
			stateIn = append(stateIn, inStates[lane])

			inRVar := &autofunc.RVariable{
				Variable:   inVar,
				ROutputVec: in.ROutputSeqs()[lane][t],
			}
			resIn = append(resIn, inRVar)
		}
		out := b.B.ApplyBlockR(rv, stateIn, resIn)
		res.StepOuts = append(res.StepOuts, out)
		outVecs := out.Outputs()
		outVecsR := out.ROutputs()
		outStates := out.RStates()
		for lane, seq := range in.OutputSeqs() {
			if len(seq) <= t {
				continue
			}
			res.Output[lane] = append(res.Output[lane], outVecs[0])
			res.ROutput[lane] = append(res.ROutput[lane], outVecsR[0])
			inStates[lane] = outStates[0]
			outVecs = outVecs[1:]
			outVecsR = outVecsR[1:]
			outStates = outStates[1:]
		}
	}
	return res
}

// Parameters returns the underlying Block's parameters
// if it implements sgd.Learner, or nil otherwise.
func (b *BlockSeqFunc) Parameters() []*autofunc.Variable {
	if l, ok := b.B.(sgd.Learner); ok {
		return l.Parameters()
	} else {
		return nil
	}
}

// SerializerType returns the unique ID used to serialize
// a BlockSeqFunc with the serializer package.
func (b *BlockSeqFunc) SerializerType() string {
	return "github.com/unixpickle/weakai/rnn.BlockSeqFunc"
}

// Serialize serializes the underlying block if it is
// a serializer.Serializer (and fails otherwise).
func (b *BlockSeqFunc) Serialize() ([]byte, error) {
	s, ok := b.B.(serializer.Serializer)
	if !ok {
		return nil, fmt.Errorf("type is not a Serializer: %T", b.B)
	}
	return serializer.SerializeWithType(s)
}

type blockSeqFuncResult struct {
	B        Block
	Start    []State
	Input    seqfunc.Result
	InPool   [][]*autofunc.Variable
	Output   [][]linalg.Vector
	StepOuts []BlockResult
}

func (b *blockSeqFuncResult) OutputSeqs() [][]linalg.Vector {
	return b.Output
}

func (b *blockSeqFuncResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	for _, poolSeq := range b.InPool {
		for _, poolVar := range poolSeq {
			g[poolVar] = make(linalg.Vector, len(poolVar.Vector))
		}
	}

	maxLen := maxSeqLength(b.Output)

	upstreamMap := map[int]StateGrad{}
	for t := maxLen - 1; t >= 0; t-- {
		var upstreamStates []StateGrad
		var upstreamVecs []linalg.Vector
		for lane, outSeq := range b.Output {
			if len(outSeq) > t {
				upstreamStates = append(upstreamStates, upstreamMap[lane])
				upstreamVecs = append(upstreamVecs, u[lane][t])
			}
		}
		downStates := b.StepOuts[t].PropagateGradient(upstreamVecs, upstreamStates, g)
		for lane, outSeq := range b.Output {
			if len(outSeq) > t {
				upstreamMap[lane] = downStates[0]
				downStates = downStates[1:]
			}
		}
	}

	startUpstream := make([]StateGrad, 0, len(upstreamMap))
	for _, x := range upstreamMap {
		startUpstream = append(startUpstream, x)
	}
	b.B.PropagateStart(b.Start, startUpstream, g)

	downstream := make([][]linalg.Vector, len(b.InPool))
	for i, poolSeq := range b.InPool {
		downstream[i] = make([]linalg.Vector, len(poolSeq))
		for j, poolVar := range poolSeq {
			downstream[i][j] = g[poolVar]
			delete(g, poolVar)
		}
	}
	b.Input.PropagateGradient(downstream, g)
}

type blockSeqFuncRResult struct {
	B        Block
	Start    []RState
	Input    seqfunc.RResult
	InPool   [][]*autofunc.Variable
	Output   [][]linalg.Vector
	ROutput  [][]linalg.Vector
	StepOuts []BlockRResult
}

func (b *blockSeqFuncRResult) OutputSeqs() [][]linalg.Vector {
	return b.Output
}

func (b *blockSeqFuncRResult) ROutputSeqs() [][]linalg.Vector {
	return b.ROutput
}

func (b *blockSeqFuncRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	if g == nil {
		g = autofunc.Gradient{}
	}

	for _, poolSeq := range b.InPool {
		for _, poolVar := range poolSeq {
			g[poolVar] = make(linalg.Vector, len(poolVar.Vector))
			rg[poolVar] = make(linalg.Vector, len(poolVar.Vector))
		}
	}

	maxLen := maxSeqLength(b.Output)

	upstreamMap := map[int]RStateGrad{}
	for t := maxLen - 1; t >= 0; t-- {
		var upstreamStates []RStateGrad
		var upstreamVecs []linalg.Vector
		var upstreamVecsR []linalg.Vector
		for lane, outSeq := range b.Output {
			if len(outSeq) > t {
				upstreamStates = append(upstreamStates, upstreamMap[lane])
				upstreamVecs = append(upstreamVecs, u[lane][t])
				upstreamVecsR = append(upstreamVecsR, uR[lane][t])
			}
		}
		downStates := b.StepOuts[t].PropagateRGradient(upstreamVecs, upstreamVecsR,
			upstreamStates, rg, g)
		for lane, outSeq := range b.Output {
			if len(outSeq) > t {
				upstreamMap[lane] = downStates[0]
				downStates = downStates[1:]
			}
		}
	}

	startUpstream := make([]RStateGrad, 0, len(upstreamMap))
	for _, x := range upstreamMap {
		startUpstream = append(startUpstream, x)
	}
	b.B.PropagateStartR(b.Start, startUpstream, rg, g)

	downstream := make([][]linalg.Vector, len(b.InPool))
	downstreamR := make([][]linalg.Vector, len(b.InPool))
	for i, poolSeq := range b.InPool {
		downstream[i] = make([]linalg.Vector, len(poolSeq))
		downstreamR[i] = make([]linalg.Vector, len(poolSeq))
		for j, poolVar := range poolSeq {
			downstream[i][j] = g[poolVar]
			downstreamR[i][j] = rg[poolVar]
			delete(g, poolVar)
			delete(rg, poolVar)
		}
	}
	b.Input.PropagateRGradient(downstream, downstreamR, rg, g)
}

func maxSeqLength(vecs [][]linalg.Vector) int {
	var max int
	for _, v := range vecs {
		if len(v) > max {
			max = len(v)
		}
	}
	return max
}
