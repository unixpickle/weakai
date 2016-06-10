package neuralnet

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
)

// A Layer represents any differentiable function
// that can be used as a layer in a neural net.
//
// Layers must be serializable, evaluatable, and
// differentiable.
//
// The autofunc.Results and autofunc.RResults from
// a Layer are only valid so long as the Layer is
// not modified (e.g. by SetCache, by changing
// a struct field, etc.).
//
// Layers must support concurrent calls to their
// autofunc.RFunc methods.
// However, serialization methods and SetCache()
// needn't be concurrency-safe--they will only
// be called if no other Layer methods are being
// called simultaneously.
type Layer interface {
	serializer.Serializer
	autofunc.RFunc

	// SetCache changes the cache that this Layer
	// will use.
	// After SetCache() is called on a layer, any of
	// the layer's previous output should be considered
	// invalid and should not be used in any way.
	SetCache(c *autofunc.VectorCache)
}

// A Randomizer is anything which can be reset to
// a random state.
// For instance, some layers of a neural network
// should be initially randomized to break symmetry.
type Randomizer interface {
	Randomize()
}

// A Learner is anything with some parameters.
type Learner interface {
	Parameters() []*autofunc.Variable
}

// A LearnBatcher is a Learner that can be evaluated
// in batch.
type BatchLearner interface {
	Learner
	MakeBatcher(c *autofunc.VectorCache) autofunc.Batcher
	MakeRBatcher(c *autofunc.VectorCache) autofunc.RBatcher
}

// A SingleLearner is a Learner that can evaluate a
// single input at once.
type SingleLearner interface {
	Learner
	autofunc.RFunc
}

// A Gradienter is anything which can compute a
// gradient for a set of inputs.
type Gradienter interface {
	// Gradient returns the total error gradient for
	// all the samples in a set.
	// The returned result is only valid until the
	// next call to Gradient (or to RGradient, if
	// this is also an RGradienter)
	Gradient(*SampleSet) autofunc.Gradient
}

// An RGradienter is anything which can compute
// a gradient and r-gradient for a set of inputs.
type RGradienter interface {
	Gradienter

	// RGradient returns the total error gradient and
	// r-gradient all the samples in a set.
	// The returned result is only valid until the
	// next call to Gradient or RGradient.
	RGradient(autofunc.RVector, *SampleSet) (autofunc.Gradient, autofunc.RGradient)
}
