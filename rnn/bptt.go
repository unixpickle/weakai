package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// BPTT is an RGradienter which uses untruncated
// back propagation through time for Sequences.
//
// After an instance of BPTT is used on a BlockLearner,
// it should never be reused on any BlockLearner with
// a different set of parameters.
type BPTT struct {
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

func (b *BPTT) Gradient(s sgd.SampleSet) autofunc.Gradient {
	return b.makeHelper().Gradient(sortSeqs(s))
}

func (b *BPTT) RGradient(v autofunc.RVector,
	s sgd.SampleSet) (autofunc.Gradient, autofunc.RGradient) {
	return b.makeHelper().RGradient(v, sortSeqs(s))
}

func (b *BPTT) makeHelper() *neuralnet.GradHelper {
	if b.helper != nil {
		b.helper.MaxConcurrency = b.MaxGoroutines
		b.helper.MaxSubBatch = b.MaxLanes
		return b.helper
	}
	b.helper = &neuralnet.GradHelper{
		MaxConcurrency: b.MaxGoroutines,
		MaxSubBatch:    b.MaxLanes,
		Learner:        b.Learner,

		CompGrad: func(g autofunc.Gradient, s sgd.SampleSet) {
			b.runBatch(nil, g, nil, sampleSetSequences(s))
		},
		CompRGrad: func(rv autofunc.RVector, rg autofunc.RGradient, g autofunc.Gradient,
			s sgd.SampleSet) {
			b.runBatch(rv, g, rg, sampleSetSequences(s))
		},
	}
	return b.helper
}

func (b *BPTT) runBatch(v autofunc.RVector, g autofunc.Gradient,
	rg autofunc.RGradient, seqs []Sequence) {
	if v == nil {
		prop := seqProp{
			Block:    b.Learner,
			CostFunc: b.CostFunc,
		}
		for len(seqs) > 0 {
			seqs = prop.TimeStep(seqs)
		}
		prop.BackPropagate(g, prop.MemoryCount(), 0)
	} else {
		prop := seqRProp{
			Block:    b.Learner,
			CostFunc: b.CostFunc,
		}
		for len(seqs) > 0 {
			seqs = prop.TimeStep(v, seqs)
		}
		prop.BackPropagate(g, rg, prop.MemoryCount(), 0)
	}
}
