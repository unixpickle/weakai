package neuralnet

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
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
// However, serialization methods needn't be safe
// for concurrency.
type Layer interface {
	serializer.Serializer
	autofunc.RFunc
}

// A Randomizer is anything which can be reset to
// a random state.
// For instance, some layers of a neural network
// should be initially randomized to break symmetry.
type Randomizer interface {
	Randomize()
}

// A LearnBatcher is a Learner that can be evaluated
// in batch.
type BatchLearner interface {
	sgd.Learner
	autofunc.RBatcher
}

// A SingleLearner is a Learner that can evaluate a
// single input at once.
type SingleLearner interface {
	sgd.Learner
	autofunc.RFunc
}
