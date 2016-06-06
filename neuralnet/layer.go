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

// A Learner is anything with parameters that should
// be learned by gradient descent or some other means
// of optimization.
type Learner interface {
	Parameters() []*autofunc.Variable
}
