package neuralnet

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// SingleRGradienter is an RGradienter that acts
// on an underlying SingleLearner to compute
// gradients for sets of BasicSamples.
//
// SingleRGradienter is good for nets with very few
// parameters, where most of the training overhead
// comes from GC overhead, function calls, etc.
//
// Once you use a SingleRGradienter with a Learner,
// you should not ever use the same SingleRGradienter
// for any learners with different parameters.
type SingleRGradienter struct {
	Learner  SingleLearner
	CostFunc CostFunc

	gradCache  autofunc.Gradient
	rgradCache autofunc.RGradient
}

func (b *SingleRGradienter) Gradient(s SampleSet) autofunc.Gradient {
	if b.gradCache == nil {
		b.gradCache = autofunc.NewGradient(b.Learner.Parameters())
	} else {
		b.gradCache.Zero()
	}

	for _, sample := range s {
		vs := sample.(VectorSample)
		output := vs.Output
		inVar := &autofunc.Variable{vs.Input}
		result := b.Learner.Apply(inVar)
		cost := b.CostFunc.Cost(output, result)
		cost.PropagateGradient(linalg.Vector{1}, b.gradCache)
	}

	return b.gradCache
}

func (b *SingleRGradienter) RGradient(rv autofunc.RVector, s SampleSet) (autofunc.Gradient,
	autofunc.RGradient) {
	if b.gradCache == nil {
		b.gradCache = autofunc.NewGradient(b.Learner.Parameters())
	} else {
		b.gradCache.Zero()
	}
	if b.rgradCache == nil {
		b.rgradCache = autofunc.NewRGradient(b.Learner.Parameters())
	} else {
		b.rgradCache.Zero()
	}

	for _, sample := range s {
		vs := sample.(VectorSample)
		output := vs.Output
		inVar := &autofunc.Variable{vs.Input}
		rVar := autofunc.NewRVariable(inVar, rv)
		result := b.Learner.ApplyR(rv, rVar)
		cost := b.CostFunc.CostR(rv, output, result)
		cost.PropagateRGradient(linalg.Vector{1}, linalg.Vector{0},
			b.rgradCache, b.gradCache)
	}

	return b.gradCache, b.rgradCache
}
