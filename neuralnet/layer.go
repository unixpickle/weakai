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

// A Learner is any RFunc with parameters that should
// be learned by gradient descent or some other means
// of optimization.
type Learner interface {
	autofunc.RFunc
	Parameters() []*autofunc.Variable
}
