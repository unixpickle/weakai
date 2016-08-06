package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// A SeqFuncGradienter computes gradients of a SeqFunc
// for sample sets full of Sequence objects.
type SeqFuncGradienter struct {
	SeqFunc  SeqFunc
	Learner  sgd.Learner
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

func (s *SeqFuncGradienter) Gradient(set sgd.SampleSet) autofunc.Gradient {
	return s.makeHelper().Gradient(sortSeqs(set))
}

func (s *SeqFuncGradienter) RGradient(v autofunc.RVector,
	set sgd.SampleSet) (autofunc.Gradient, autofunc.RGradient) {
	return s.makeHelper().RGradient(v, sortSeqs(set))
}

func (s *SeqFuncGradienter) makeHelper() *neuralnet.GradHelper {
	if s.helper != nil {
		s.helper.MaxConcurrency = s.MaxGoroutines
		s.helper.MaxSubBatch = s.MaxLanes
		return s.helper
	}
	s.helper = &neuralnet.GradHelper{
		MaxConcurrency: s.MaxGoroutines,
		MaxSubBatch:    s.MaxLanes,
		Learner:        s.Learner,
		CompGrad:       s.runBatch,
		CompRGrad:      s.runBatchR,
	}
	return s.helper
}

func (s *SeqFuncGradienter) runBatch(g autofunc.Gradient, set sgd.SampleSet) {
	seqs := sampleSetSequences(set)
	seqIns := make([][]autofunc.Result, len(seqs))
	for i, seq := range seqs {
		ins := make([]autofunc.Result, len(seq.Inputs))
		for j, x := range seq.Inputs {
			ins[j] = &autofunc.Variable{Vector: x}
		}
		seqIns[i] = ins
	}

	output := s.SeqFunc.BatchSeqs(seqIns)

	upstream := make([][]linalg.Vector, len(seqIns))
	for i, outSeq := range output.OutputSeqs() {
		us := make([]linalg.Vector, len(outSeq))
		expectedSeq := seqs[i].Outputs
		for j, actual := range outSeq {
			expected := expectedSeq[j]
			us[j] = costFuncDeriv(s.CostFunc, expected, actual)
		}
		upstream[i] = us
	}

	output.Gradient(upstream, g)
}

func (s *SeqFuncGradienter) runBatchR(rv autofunc.RVector, rg autofunc.RGradient,
	g autofunc.Gradient, set sgd.SampleSet) {
	seqs := sampleSetSequences(set)
	seqIns := make([][]autofunc.RResult, len(seqs))
	var zeroVec linalg.Vector
	for i, seq := range seqs {
		ins := make([]autofunc.RResult, len(seq.Inputs))
		for j, x := range seq.Inputs {
			variable := &autofunc.Variable{Vector: x}
			if zeroVec == nil {
				zeroVec = make(linalg.Vector, len(x))
			}
			ins[j] = &autofunc.RVariable{
				Variable:   variable,
				ROutputVec: zeroVec,
			}
		}
		seqIns[i] = ins
	}

	output := s.SeqFunc.BatchSeqsR(rv, seqIns)

	upstream := make([][]linalg.Vector, len(seqIns))
	upstreamR := make([][]linalg.Vector, len(seqIns))
	for i, outSeq := range output.OutputSeqs() {
		rOutSeq := output.ROutputSeqs()[i]
		us := make([]linalg.Vector, len(outSeq))
		usR := make([]linalg.Vector, len(outSeq))
		expectedSeq := seqs[i].Outputs
		for j, actual := range outSeq {
			expected := expectedSeq[j]
			us[j], usR[j] = costFuncRDeriv(s.CostFunc, expected, actual, rOutSeq[j])
		}
		upstream[i] = us
		upstreamR[i] = usR
	}

	output.RGradient(upstream, upstreamR, rg, g)
}
