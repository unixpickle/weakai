package convnet

import "github.com/unixpickle/num-analysis/kahan"

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

	// WeightGradients has the same structure as
	// Weights, and each entry corresponds to the
	// partial derivative of the loss function with
	// respect to the given weight.
	// It will be setup during backward-propagation.
	WeightGradients [][]float64

	// BiasGradients is like WeightGradients, but for
	// Biases instead of Weights.
	BiasGradients []float64

	// UpstreamGradients has the same structure as
	// Inputs, and each entry corresponds to the
	// partial derivative of the loss function with
	// respect to the given input.
	// It will be setup during backward-propagation.
	UpstreamGradients []float64

	// Input is an array of input values
	// for this layer.
	// This should be setup by an external entity
	// before forward-propagation.
	Input []float64

	// DownstreamGradients is structured like Output,
	// and each entry corresponds to the partial of
	// the loss function with respect to the output
	// from this layer.
	// This should be setup  by an external entity
	// before backward-propagation.
	DownstreamGradients []float64

	outputSums []float64
}

func NewDenseLayer(params *DenseParams) *DenseLayer {
	res := &DenseLayer{
		Activation:        params.Activation,
		Weights:           make([][]float64, params.OutputCount),
		Biases:            make([]float64, params.OutputCount),
		Output:            make([]float64, params.OutputCount),
		WeightGradients:   make([][]float64, params.OutputCount),
		BiasGradients:     make([]float64, params.OutputCount),
		UpstreamGradients: make([]float64, params.OutputCount),
		outputSums:        make([]float64, params.OutputCount),
	}
	for i := range res.Weights {
		res.Weights[i] = make([]float64, params.InputCount)
		res.WeightGradients[i] = make([]float64, params.InputCount)
	}
	return res
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
	for i := range d.UpstreamGradients {
		d.UpstreamGradients[i] = 0
	}

	for i, weights := range d.Weights {
		sumPartial := d.DownstreamGradients[i] * d.Activation.Deriv(d.outputSums[i])
		d.BiasGradients[i] = sumPartial
		for j, weight := range weights {
			d.WeightGradients[i][j] = d.Input[j] * sumPartial
			d.UpstreamGradients[j] += sumPartial * weight
		}
	}
}
