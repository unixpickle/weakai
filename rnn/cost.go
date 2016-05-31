package rnn

type CostFunc interface {
	// Gradient computes the gradient of the cost
	// function and writes it to gradient.
	// It uses the actual and expected outputs to
	// compute the cost.
	Gradient(actual, expected, gradient []float64)
}

type MeanSquaredCost struct{}

func (_ MeanSquaredCost) Gradient(actual, expected, gradient []float64) {
	for i, x := range expected {
		gradient[i] = actual[i] - x
	}
}
