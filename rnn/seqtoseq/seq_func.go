package seqtoseq

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// A Gradienter is an sgd.RGradienter which works on a
// seqfunc.RFunc for sample sets of Sequence objects.
//
// After an instance is used once, it should never be
// reused with different parameters.
type SeqFuncGradienter struct {
	SeqFunc  seqfunc.RFunc
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
	seqs := sampleSetSlice(set)
	var seqIns [][]linalg.Vector
	for _, s := range seqs {
		seqIns = append(seqIns, s.Inputs)
	}
	output := s.SeqFunc.ApplySeqs(seqfunc.ConstResult(seqIns))

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

	output.PropagateGradient(upstream, g)
}

func (s *SeqFuncGradienter) runBatchR(rv autofunc.RVector, rg autofunc.RGradient,
	g autofunc.Gradient, set sgd.SampleSet) {
	seqs := sampleSetSlice(set)
	seqIns := make([][]linalg.Vector, len(seqs))
	for _, s := range seqs {
		seqIns = append(seqIns, s.Inputs)
	}
	output := s.SeqFunc.ApplySeqsR(rv, seqfunc.ConstRResult(seqIns))

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

	output.PropagateRGradient(upstream, upstreamR, rg, g)
}
