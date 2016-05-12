package neuralnet

// A CostFunc computes the amount of
// discrepancy between the actual and
// expected output of a function.
//
// A cost function itself needn't be
// computable, as long as its gradient can
// be computed with respect to the actual
// outputs.
type CostFunc interface {
	// Deriv computes the gradient of the
	// loss function, given the actual and
	// expected outputs.
	// The result is written to gradOut.
	Deriv(gradOut, actual, expected []float64)
}

type MeanSquaredCost struct{}

func (_ MeanSquaredCost) Deriv(gradOut, actual, expected []float64) {
	for i, x := range actual {
		gradOut[i] = x - expected[i]
	}
}
