package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// BatcherBlock turns an autofunc.RBatcher into a Block.
//
// The inputs and outputs of the RBatcher are packed by
// appending the state to the input/output.
// For example, an input of {1,2} and a state of {3,4}
// would be packed as {1,2,3,4}.
type BatcherBlock struct {
	F            autofunc.RBatcher
	StateSizeVal int
}

func (b *BatcherBlock) StateSize() int {
	return b.StateSizeVal
}

func (f *BatcherBlock) Batch(in *BlockInput) BlockOutput {
	results := make([]autofunc.Result, 0, len(in.States)*2)
	for i := range in.States {
		results = append(results, in.Inputs[i], in.States[i])
	}
	joined := autofunc.Concat(results...)
	output := f.F.Batch(joined, len(in.States))
	return &batcherBlockOutput{
		Result:    output,
		StateSize: f.StateSizeVal,
		LaneCount: len(in.States),
	}
}

func (f *BatcherBlock) BatchR(v autofunc.RVector, in *BlockRInput) BlockROutput {
	results := make([]autofunc.RResult, 0, len(in.States)*2)
	for i := range in.States {
		results = append(results, in.Inputs[i], in.States[i])
	}
	joined := autofunc.ConcatR(results...)
	output := f.F.BatchR(v, joined, len(in.States))
	return &batcherBlockOutput{
		RResult:   output,
		StateSize: f.StateSizeVal,
		LaneCount: len(in.States),
	}
}

type batcherBlockOutput struct {
	Result    autofunc.Result
	RResult   autofunc.RResult
	StateSize int
	LaneCount int
}

func (b *batcherBlockOutput) States() []linalg.Vector {
	return b.extractStates(b.resultOutputVector())
}

func (b *batcherBlockOutput) RStates() []linalg.Vector {
	return b.extractStates(b.RResult.ROutput())
}

func (b *batcherBlockOutput) Outputs() []linalg.Vector {
	return b.extractOutputs(b.resultOutputVector())
}

func (b *batcherBlockOutput) ROutputs() []linalg.Vector {
	return b.extractOutputs(b.RResult.ROutput())
}

func (b *batcherBlockOutput) Gradient(u *UpstreamGradient, g autofunc.Gradient) {
	upstreamVec := b.joinUpstream(u.States, u.Outputs)
	b.Result.PropagateGradient(upstreamVec, g)
}

func (b *batcherBlockOutput) RGradient(u *UpstreamRGradient, rg autofunc.RGradient,
	g autofunc.Gradient) {
	upstreamVec := b.joinUpstream(u.States, u.Outputs)
	rupstreamVec := b.joinUpstream(u.RStates, u.ROutputs)
	b.RResult.PropagateRGradient(upstreamVec, rupstreamVec, rg, g)
}

func (b *batcherBlockOutput) extractStates(output linalg.Vector) []linalg.Vector {
	outSize := b.outputSize()
	res := make([]linalg.Vector, b.LaneCount)
	for i := range res {
		startIdx := i*(outSize+b.StateSize) + outSize
		res[i] = output[startIdx : startIdx+b.StateSize]
	}
	return res
}

func (b *batcherBlockOutput) extractOutputs(output linalg.Vector) []linalg.Vector {
	outSize := b.outputSize()
	res := make([]linalg.Vector, b.LaneCount)
	for i := range res {
		startIdx := i * (outSize + b.StateSize)
		res[i] = output[startIdx : startIdx+outSize]
	}
	return res
}

func (b *batcherBlockOutput) joinUpstream(states, outputs []linalg.Vector) linalg.Vector {
	upstreamVec := make(linalg.Vector, (len(states[0])+len(outputs[0]))*len(states))
	var idx int
	for i, output := range outputs {
		copy(upstreamVec[idx:], output)
		idx += len(output)
		state := states[i]
		copy(upstreamVec[idx:], state)
		idx += len(state)
	}
	return upstreamVec
}

func (b *batcherBlockOutput) outputSize() int {
	return len(b.resultOutputVector())/b.LaneCount - b.StateSize
}

func (b *batcherBlockOutput) resultOutputVector() linalg.Vector {
	if b.Result != nil {
		return b.Result.Output()
	} else {
		return b.RResult.Output()
	}
}
