package convnet

import (
	"math/rand"

	"github.com/unixpickle/num-analysis/kahan"
)

// DenseParams are parameters for a dense
// or "fully-connected" layer.
type DenseParams struct {
	Activation  ActivationFunc
	InputCount  int
	OutputCount int
}

type DenseLayer struct {
	Activation ActivationFunc

	// Weights is a slice of weight maps, where
	// each weight map corresponds to a neuron
	// in the layer.
	Weights [][]float64

	// Biases is a slice of biases, one for each
	// output neuron.
	Biases []float64

	// Output is an array of output values
	// from this layer.
	// It will be set during forward-propagation.
	Output []float64

	// WeightGradient has the same structure as
	// Weights, and each entry corresponds to the
	// partial derivative of the cost function with
	// respect to the given weight.
	// It will be setup during backward-propagation.
	WeightGradient [][]float64

	// BiasGradient is like WeightGradient, but for
	// Biases instead of Weights.
	BiasGradient []float64

	// UpstreamGradient has the same structure as
	// Inputs, and each entry corresponds to the
	// partial derivative of the cost function with
	// respect to the given input.
	// It will be setup during backward-propagation.
	UpstreamGradient []float64

	// Input is an array of input values
	// for this layer.
	// This should be setup by an external entity
	// before forward-propagation.
	Input []float64

	// DownstreamGradient is structured like Output,
	// and each entry corresponds to the partial of
	// the cost function with respect to the output
	// from this layer.
	// This should be setup  by an external entity
	// before backward-propagation.
	DownstreamGradient []float64

	outputSums []float64
}

func NewDenseLayer(params *DenseParams) *DenseLayer {
	res := &DenseLayer{
		Activation:       params.Activation,
		Weights:          make([][]float64, params.OutputCount),
		Biases:           make([]float64, params.OutputCount),
		Output:           make([]float64, params.OutputCount),
		WeightGradient:   make([][]float64, params.OutputCount),
		BiasGradient:     make([]float64, params.OutputCount),
		UpstreamGradient: make([]float64, params.OutputCount),
		outputSums:       make([]float64, params.OutputCount),
	}
	for i := range res.Weights {
		res.Weights[i] = make([]float64, params.InputCount)
		res.WeightGradient[i] = make([]float64, params.InputCount)
	}
	return res
}

func (d *DenseLayer) Randomize() {
	for i := range d.Biases {
		d.Biases[i] = (rand.Float64() * 2) - 1
	}
	for _, weights := range d.Weights {
		for i := range weights {
			weights[i] = (rand.Float64() * 2) - 1
		}
	}
}

// PropagateForward performs forward-propagation.
func (d *DenseLayer) PropagateForward() {
	for i, weights := range d.Weights {
		sum := kahan.NewSummer64()
		for j, weight := range weights {
			sum.Add(weight * d.Input[j])
		}
		sum.Add(d.Biases[i])
		d.outputSums[i] = sum.Sum()
		d.Output[i] = d.Activation.Eval(sum.Sum())
	}
}

// PropagateBackward performs backward-propagation.
func (d *DenseLayer) PropagateBackward() {
	for i := range d.UpstreamGradient {
		d.UpstreamGradient[i] = 0
	}

	for i, weights := range d.Weights {
		sumPartial := d.DownstreamGradient[i] * d.Activation.Deriv(d.outputSums[i])
		d.BiasGradient[i] = sumPartial
		for j, weight := range weights {
			d.WeightGradient[i][j] = d.Input[j] * sumPartial
			d.UpstreamGradient[j] += sumPartial * weight
		}
	}
}
