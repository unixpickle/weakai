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
