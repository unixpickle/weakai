package neuralnet

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
)

// BatchRGradienter is an RGradienter that computes
// the gradients of BatchLearners using SampleSets
// of VectorSamples.
//
// A BatchRGradienter is suitable for training tasks
// with lots of learnable parameters, since said tasks
// can benefit from parallelism.
//
// After you use a BatchRGradienter with a given
// BatchLearner, you should never use the same
// BatchRGradienter for any BatchLearner with
// different parameters.
type BatchRGradienter struct {
	Learner  BatchLearner
	CostFunc CostFunc

	// MaxGoroutines is the maximum number of Goroutines
	// the BatchRGradienter will use simultaneously.
	// If this is 0, a reasonable default is used.
	MaxGoroutines int

	// MaxBatchSize is the maximum number of samples the
	// BatchRGradienter will pass to the learner at once.
	// If this is 0, a reasonable default is used.
	MaxBatchSize int

	helper *GradHelper
}

func (b *BatchRGradienter) Gradient(s sgd.SampleSet) autofunc.Gradient {
	return b.makeHelper().Gradient(s)
}

func (b *BatchRGradienter) RGradient(v autofunc.RVector, s sgd.SampleSet) (autofunc.Gradient,
	autofunc.RGradient) {
	return b.makeHelper().RGradient(v, s)
}

func (b *BatchRGradienter) makeHelper() *GradHelper {
	if b.helper != nil {
		b.helper.MaxConcurrency = b.MaxGoroutines
		b.helper.MaxSubBatch = b.MaxBatchSize
		return b.helper
	}
	b.helper = &GradHelper{
		MaxConcurrency: b.MaxGoroutines,
		MaxSubBatch:    b.MaxBatchSize,
		Learner:        b.Learner,

		CompGrad: func(g autofunc.Gradient, s sgd.SampleSet) {
			b.runBatch(nil, nil, g, s)
		},
		CompRGrad: b.runBatch,
	}
	return b.helper
}

func (b *BatchRGradienter) runBatch(rv autofunc.RVector, rgrad autofunc.RGradient,
	grad autofunc.Gradient, s sgd.SampleSet) {
	if s.Len() == 0 {
		return
	}

	sampleCount := s.Len()
	firstSample := s.GetSample(0).(VectorSample)
	inputSize := len(firstSample.Input)
	outputSize := len(firstSample.Output)
	inVec := make(linalg.Vector, sampleCount*inputSize)
	outVec := make(linalg.Vector, sampleCount*outputSize)

	for i := 0; i < s.Len(); i++ {
		sample := s.GetSample(i)
		vs := sample.(VectorSample)
		copy(inVec[i*inputSize:], vs.Input)
		copy(outVec[i*outputSize:], vs.Output)
	}

	inVar := &autofunc.Variable{inVec}
	if rgrad != nil {
		rVar := autofunc.NewRVariable(inVar, rv)
		result := b.Learner.BatchR(rv, rVar, sampleCount)
		cost := b.CostFunc.CostR(rv, outVec, result)
		cost.PropagateRGradient(linalg.Vector{1}, linalg.Vector{0},
			rgrad, grad)
	} else {
		result := b.Learner.Batch(inVar, sampleCount)
		cost := b.CostFunc.Cost(outVec, result)
		cost.PropagateGradient(linalg.Vector{1}, grad)
	}
}
