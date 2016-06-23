package rnn

import (
	"sort"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

// FullRGradienter is an RGradienter which computes
// untruncated gradients for sets of Sequences.
//
// After a FullRGradienter is used on a BlockLearner,
// it should never be reused on any BlockLearner with
// a different set of parameters.
type FullRGradienter struct {
	Learner  BlockLearner
	CostFunc neuralnet.CostFunc

	// MaxLanes specifies the maximum number of lanes
	// any BlockInput or BlockRInput may have at once
	// while computing gradients.
	// If this is 0, a reasonable default is used.
	MaxLanes int

	// MaxGoroutines specifies the maximum number of
	// Goroutines on which to invoke Batch or BatchR
	// on Learner at once.
	MaxGoroutines int

	helper *neuralnet.GradHelper
}

func (b *FullRGradienter) Gradient(s neuralnet.SampleSet) autofunc.Gradient {
	return b.makeHelper().Gradient(sortSeqs(s))
}

func (b *FullRGradienter) RGradient(v autofunc.RVector,
	s neuralnet.SampleSet) (autofunc.Gradient, autofunc.RGradient) {
	return b.makeHelper().RGradient(v, sortSeqs(s))
}

func (b *FullRGradienter) makeHelper() *neuralnet.GradHelper {
	if b.helper != nil {
		b.helper.MaxConcurrency = b.MaxGoroutines
		b.helper.MaxSubBatch = b.MaxLanes
		return b.helper
	}
	b.helper = &neuralnet.GradHelper{
		MaxConcurrency: b.MaxGoroutines,
		MaxSubBatch:    b.MaxLanes,
		Learner:        b.Learner,

		CompGrad: func(g autofunc.Gradient, s neuralnet.SampleSet) {
			b.runBatch(nil, g, nil, sampleSetSequences(s))
		},
		CompRGrad: func(rv autofunc.RVector, rg autofunc.RGradient, g autofunc.Gradient,
			s neuralnet.SampleSet) {
			b.runBatch(rv, g, rg, sampleSetSequences(s))
		},
	}
	return b.helper
}

func (b *FullRGradienter) runBatch(v autofunc.RVector, g autofunc.Gradient,
	rg autofunc.RGradient, seqs []Sequence) {
	seqs = removeEmpty(seqs)
	if len(seqs) == 0 {
		return
	}

	emptyState := make(linalg.Vector, b.Learner.StateSize())
	zeroStates := make([]linalg.Vector, len(seqs))
	for i := range zeroStates {
		zeroStates[i] = emptyState
	}

	if v != nil {
		b.recursiveRBatch(v, g, rg, seqs, zeroStates, zeroStates)
	} else {
		b.recursiveBatch(g, seqs, zeroStates)
	}
}

func (b *FullRGradienter) recursiveBatch(g autofunc.Gradient, seqs []Sequence,
	lastStates []linalg.Vector) []linalg.Vector {
	input := seqHeadInput(seqs, lastStates)
	res := b.Learner.Batch(input)

	// Compute upstream state derivatives recursively.
	upstream := &UpstreamGradient{}
	nextSeqs := removeFirst(seqs)
	if len(nextSeqs) != 0 {
		nextStates := filterContinued(seqs, res.States())
		res := b.recursiveBatch(g, nextSeqs, nextStates)
		upstream.States = injectDiscontinued(seqs, res, b.Learner.StateSize())
	}

	for lane, output := range res.Outputs() {
		outGrad := costFuncDeriv(b.CostFunc, seqs[lane].Outputs[0], output)
		upstream.Outputs = append(upstream.Outputs, outGrad)
	}

	// Compute downstream state derivatives & back propagate.
	downstream := make([]linalg.Vector, len(input.States))
	for i, s := range input.States {
		downstream[i] = make(linalg.Vector, b.Learner.StateSize())
		g[s] = downstream[i]
	}
	res.Gradient(upstream, g)
	for _, s := range input.States {
		delete(g, s)
	}
	return downstream
}

func (b *FullRGradienter) recursiveRBatch(v autofunc.RVector, g autofunc.Gradient,
	rg autofunc.RGradient, seqs []Sequence, states, rStates []linalg.Vector) (stateGrad,
	stateRGrad []linalg.Vector) {
	input := seqHeadRInput(seqs, states, rStates)
	res := b.Learner.BatchR(v, input)

	// Compute upstream state derivatives recursively.
	upstream := &UpstreamRGradient{}
	nextSeqs := removeFirst(seqs)
	if len(nextSeqs) != 0 {
		nextStates := filterContinued(seqs, res.States())
		nextRStates := filterContinued(seqs, res.RStates())
		states, statesR := b.recursiveRBatch(v, g, rg, nextSeqs, nextStates, nextRStates)
		upstream.States = injectDiscontinued(seqs, states, b.Learner.StateSize())
		upstream.RStates = injectDiscontinued(seqs, statesR, b.Learner.StateSize())
	}

	for lane, output := range res.Outputs() {
		rOutput := res.ROutputs()[lane]
		outGrad, outRGrad := costFuncRDeriv(v, b.CostFunc, seqs[lane].Outputs[0],
			output, rOutput)
		upstream.Outputs = append(upstream.Outputs, outGrad)
		upstream.ROutputs = append(upstream.ROutputs, outRGrad)
	}

	// Compute downstream state derivatives & back propagate.
	stateGrad = make([]linalg.Vector, len(input.States))
	stateRGrad = make([]linalg.Vector, len(input.States))
	for i, s := range input.States {
		stateGrad[i] = make(linalg.Vector, b.Learner.StateSize())
		stateRGrad[i] = make(linalg.Vector, b.Learner.StateSize())
		g[s.Variable] = stateGrad[i]
		rg[s.Variable] = stateRGrad[i]
	}
	res.RGradient(upstream, rg, g)
	for _, s := range input.States {
		delete(g, s.Variable)
		delete(rg, s.Variable)
	}
	return
}

func removeEmpty(seqs []Sequence) []Sequence {
	var res []Sequence
	for _, seq := range seqs {
		if len(seq.Inputs) != 0 {
			res = append(res, seq)
		}
	}
	return res
}

func removeFirst(seqs []Sequence) []Sequence {
	var nextSeqs []Sequence
	for _, seq := range seqs {
		if len(seq.Inputs) == 1 {
			continue
		}
		s := Sequence{Inputs: seq.Inputs[1:], Outputs: seq.Outputs[1:]}
		nextSeqs = append(nextSeqs, s)
	}
	return nextSeqs
}

// filterContinued filters ins so that input i
// is only kept if the i-th sequence has more
// than one input in it.
func filterContinued(seqs []Sequence, ins []linalg.Vector) []linalg.Vector {
	var res []linalg.Vector
	for i, seq := range seqs {
		if len(seq.Inputs) > 1 {
			res = append(res, ins[i])
		}
	}
	return res
}

// injectDiscontinued injects zeroed slices in
// a result for every element of seqs which has
// less than two inputs.
func injectDiscontinued(seqs []Sequence, outs []linalg.Vector, vecLen int) []linalg.Vector {
	var zeroVec linalg.Vector
	var res []linalg.Vector
	var takeIdx int
	for _, s := range seqs {
		if len(s.Inputs) > 1 {
			res = append(res, outs[takeIdx])
			takeIdx++
		} else {
			if zeroVec == nil {
				zeroVec = make(linalg.Vector, vecLen)
			}
			res = append(res, zeroVec)
		}
	}
	return res
}

func sortSeqs(s neuralnet.SampleSet) neuralnet.SampleSet {
	origSet := sampleSetSequences(s)
	res := make(seqSorter, len(origSet))
	copy(res, origSet)
	sort.Sort(res)

	resSet := make(neuralnet.SliceSampleSet, len(res))
	for i, x := range res {
		resSet[i] = x
	}
	return resSet
}

type seqSorter []Sequence

func (s seqSorter) Len() int {
	return len(s)
}

func (s seqSorter) Less(i, j int) bool {
	return len(s[i].Inputs) > len(s[j].Inputs)
}

func (s seqSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
