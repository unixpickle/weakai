package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// TruncatedBPTT is an RGradienter which uses truncated
// back propagation through time.
//
// For traditional truncated BPTT, use a HeadSize of
// 1 and a TailSize of the desired number of timesteps
// through which to back-propagate.
//
// Truncated BPTT makes it possible to avoid exploding
// gradients while still covering relatively large
// chronological gaps.
// The idea is to break up long sequences into chunks
// of size HeadSize (e.g. 1, for regular TBPTT).
// However, instead of training each of these smaller
// chunks as separate sequences, the chunks are trained
// with the state output from the previous chunk.
// Errors are then propagate backwards through this state
// for TailSize time steps.
// This way, BPTT is applied to smaller chunks, yet the
// network still learns to create states that work well
// across chunks.
type TruncatedBPTT struct {
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

	HeadSize int
	TailSize int

	helper *neuralnet.GradHelper
}

func (t *TruncatedBPTT) Gradient(s sgd.SampleSet) autofunc.Gradient {
	return t.makeHelper().Gradient(sortSeqs(s))
}

func (t *TruncatedBPTT) RGradient(v autofunc.RVector,
	s sgd.SampleSet) (autofunc.Gradient, autofunc.RGradient) {
	return t.makeHelper().RGradient(v, sortSeqs(s))
}

func (t *TruncatedBPTT) makeHelper() *neuralnet.GradHelper {
	if t.helper != nil {
		t.helper.MaxConcurrency = t.MaxGoroutines
		t.helper.MaxSubBatch = t.MaxLanes
		return t.helper
	}
	t.helper = &neuralnet.GradHelper{
		MaxConcurrency: t.MaxGoroutines,
		MaxSubBatch:    t.MaxLanes,
		Learner:        t.Learner,

		CompGrad: func(g autofunc.Gradient, s sgd.SampleSet) {
			t.runBatch(nil, g, nil, sampleSetSequences(s))
		},
		CompRGrad: func(rv autofunc.RVector, rg autofunc.RGradient, g autofunc.Gradient,
			s sgd.SampleSet) {
			t.runBatch(rv, g, rg, sampleSetSequences(s))
		},
	}
	return t.helper
}

func (t *TruncatedBPTT) runBatch(v autofunc.RVector, g autofunc.Gradient,
	rg autofunc.RGradient, seqs []Sequence) {
	if v == nil {
		prop := seqProp{
			Block:    t.Learner,
			CostFunc: t.CostFunc,
		}
		headLen := 0
		for len(seqs) > 0 {
			seqs = prop.TimeStep(seqs)
			headLen++
			if headLen == t.HeadSize || len(seqs) == 0 {
				prop.BackPropagate(g, headLen, t.TailSize)
				headLen = 0
				prop.Truncate(t.TailSize)
			}
		}
	} else {
		prop := seqRProp{
			Block:    t.Learner,
			CostFunc: t.CostFunc,
		}
		headLen := 0
		for len(seqs) > 0 {
			seqs = prop.TimeStep(v, seqs)
			headLen++
			if headLen == t.HeadSize || len(seqs) == 0 {
				prop.BackPropagate(g, rg, headLen, t.TailSize)
				headLen = 0
				prop.Truncate(t.TailSize)
			}
		}
	}
}
