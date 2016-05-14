package neuralnet

// A CostFunc computes some metric of the
// "error" for the result of a Layer.
//
// A cost function itself needn't be
// computable, as long as its gradient can
// be computed with respect to the actual
// outputs.
type CostFunc interface {
	// Deriv computes the gradient of the
	// cost function, given the layer whose
	// output should be analyzed and the
	// expected output.
	// The result is written to gradOut.
	Deriv(layer Layer, expected, gradOut []float64)

	// UpdateInternal should be called
	// after back propagation but before
	// gradient descent.
	// It allows the cost function to update
	// the internal gradients of the layer
	// to reflect the cost's dependence on
	// the internal parameters.
	// For a cost function which performs no
	// regularization, this is a no-op.
	UpdateInternal(Layer)
}

type MeanSquaredCost struct{}

func (_ MeanSquaredCost) Deriv(layer Layer, expected, gradOut []float64) {
	actual := layer.Output()
	for i, x := range actual {
		gradOut[i] = x - expected[i]
	}
}

func (_ MeanSquaredCost) UpdateInternal(Layer) {
}

type CrossEntropyCost struct{}

func (_ CrossEntropyCost) Deriv(layer Layer, expected, gradOut []float64) {
	actual := layer.Output()
	for i, x := range expected {
		a := actual[i]
		gradOut[i] = (a - x) / ((a + 1) * a)
	}
}

func (_ CrossEntropyCost) UpdateInternal(Layer) {
}

// SparseRegularizingCost wraps another cost
// function and adds the squares of every
// weight and bias of every ConvLayer,
// DenseLayer, and ConvGrowLayer.
type SparseRegularizingCost struct {
	Cost CostFunc

	BiasPenalty   float64
	WeightPenalty float64
}

func (r SparseRegularizingCost) Deriv(layer Layer, expected, gradOut []float64) {
	r.Cost.Deriv(layer, expected, gradOut)
}

func (r SparseRegularizingCost) UpdateInternal(layer Layer) {
	switch layer := layer.(type) {
	case *Network:
		for _, subLayer := range layer.Layers {
			r.UpdateInternal(subLayer)
		}
	case *ConvLayer:
		filterGrads := layer.FilterGradients()
		filterWeights := layer.Filters()
		for i, weights := range filterWeights {
			grads := filterGrads[i]
			for j, w := range weights.Data {
				grads.Data[j] += w * r.WeightPenalty
			}
		}

		biases := layer.Biases()
		biasGrads := layer.BiasGradients()
		for i, bias := range biases {
			biasGrads[i] += bias * r.BiasPenalty
		}
	case *DenseLayer:
		weights := layer.Weights()
		gradients := layer.WeightGradients()
		for i, neuron := range weights {
			neuronGrad := gradients[i]
			for j, w := range neuron {
				neuronGrad[j] += w * r.WeightPenalty
			}
		}

		biases := layer.Biases()
		biasGrads := layer.BiasGradients()
		for i, bias := range biases {
			biasGrads[i] += bias * r.BiasPenalty
		}
	case *ConvGrowLayer:
		r.UpdateInternal(layer.ConvLayer())
	}
}
