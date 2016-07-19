// Package neuralnet provides APIs for implementing and
// training feedforward neural networks.
//
// Every layer in a neural network implements the Layer
// interface. Any set of Layers can be stacked in a
// Network (a type equivalent to []Layer), which is
// itself a Layer. All Layers can be serialized, so
// saving a neural network to a file is easy.
//
// To train a network, you can use the sgd package at
// https://github.com/unixpickle/sgd or the Hessian Free
// package at https://github.com/unixpickle/hessfree.
// Both packages depend on the sgd.Gradienter interface,
// which is implemented by BatchRGradienter and
// SingleRGradienter.
//
// Here is how you could train a simple neural network:
//
//     var trainingSamples, trainingOutputs []linalg.Vector
//     ...
//     samples := neuralnet.VectorSampleSet(trainingSamples, trainingOutputs)
//
//     network := neuralnet.Network{
//         &neuralnet.DenseLayer{
//             InputCount:  InputVectorSize,
//             OutputCount: FirstHiddenSize,
//         },
//         &neuralnet.Sigmoid{},
//         &neuralnet.DenseLayer{
//             InputCount:  FirstHiddenSize,
//             OutputCount: OutputSize,
//         },
//         &neuralnet.Sigmoid{},
//     }
//     network.Randomize()
//
//     batcher := &neuralnet.BatchRGradienter{
//         Learner:  network.BatchLearner(),
//         CostFunc: neuralnet.MeanSquaredCost{},
//     }
//     sgd.SGD(batcher, samples, 0.2, 100000, 1)
//
package neuralnet

import (
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
)

// Network is a Layer formed by composing many
// other layers.
// It also facilitates aggregate operations on
// the underlying layers through its Randomize()
// and Parameters() methods.
type Network []Layer

func DeserializeNetwork(data []byte) (Network, error) {
	var res Network

	slice, err := serializer.DeserializeSlice(data)
	if err != nil {
		return nil, err
	}

	for _, x := range slice {
		if layer, ok := x.(Layer); ok {
			res = append(res, layer)
		} else {
			return nil, errors.New("slice element is not a Layer")
		}
	}

	return res, nil
}

// Randomize calls Randomize() on all the layers
// in n that implement Randomizer.
func (n Network) Randomize() {
	for _, layer := range n {
		if r, ok := layer.(Randomizer); ok {
			r.Randomize()
		}
	}
}

// Parameters concatenates the parameters of
// every Learner in n.
func (n Network) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, layer := range n {
		if l, ok := layer.(sgd.Learner); ok {
			res = append(res, l.Parameters()...)
		}
	}
	return res
}

func (n Network) Apply(in autofunc.Result) autofunc.Result {
	for _, layer := range n {
		in = layer.Apply(in)
	}
	return in
}

func (n Network) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	for _, layer := range n {
		in = layer.ApplyR(v, in)
	}
	return in
}

func (n Network) Serialize() ([]byte, error) {
	serializers := make([]serializer.Serializer, len(n))
	for i, x := range n {
		serializers[i] = x
	}
	return serializer.SerializeSlice(serializers)
}

func (n Network) SerializerType() string {
	return serializerTypeNetwork
}

// BatchLearner creates a new BatchLearner that is
// based on n.
// If you modify the network, all BatchLearners made
// from it should be considered invalid.
func (n Network) BatchLearner() BatchLearner {
	return &networkBatchLearner{
		Batcher:  n.makeBatcher(),
		RBatcher: n.makeRBatcher(),
		Network:  n,
	}
}

func (n Network) makeBatcher() autofunc.Batcher {
	var currentFunc autofunc.ComposedFunc
	var result autofunc.ComposedBatcher
	for _, layer := range n {
		if b, ok := layer.(autofunc.Batcher); ok {
			if len(currentFunc) != 0 {
				fb := &autofunc.FuncBatcher{F: currentFunc}
				result = append(result, fb)
				currentFunc = nil
			}
			result = append(result, b)
		} else {
			currentFunc = append(currentFunc, layer)
		}
	}
	if len(currentFunc) != 0 {
		fb := &autofunc.FuncBatcher{F: currentFunc}
		result = append(result, fb)
	}
	return result
}

func (n Network) makeRBatcher() autofunc.RBatcher {
	var currentFunc autofunc.ComposedRFunc
	var result autofunc.ComposedRBatcher
	for _, layer := range n {
		if b, ok := layer.(autofunc.RBatcher); ok {
			if len(currentFunc) != 0 {
				fb := &autofunc.RFuncBatcher{F: currentFunc}
				result = append(result, fb)
				currentFunc = nil
			}
			result = append(result, b)
		} else {
			currentFunc = append(currentFunc, layer)
		}
	}
	if len(currentFunc) != 0 {
		fb := &autofunc.RFuncBatcher{F: currentFunc}
		result = append(result, fb)
	}
	return result
}

type networkBatchLearner struct {
	Batcher  autofunc.Batcher
	RBatcher autofunc.RBatcher
	Network  Network
}

func (n *networkBatchLearner) Batch(in autofunc.Result, m int) autofunc.Result {
	return n.Batcher.Batch(in, m)
}

func (n *networkBatchLearner) BatchR(v autofunc.RVector, in autofunc.RResult,
	m int) autofunc.RResult {
	return n.RBatcher.BatchR(v, in, m)
}

func (n *networkBatchLearner) Parameters() []*autofunc.Variable {
	return n.Network.Parameters()
}
