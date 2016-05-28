package lstm

type CostFunc interface {
	Gradient(actual, expected, gradient []float64)
}

type MeanSquaredCost struct{}

func (_ MeanSquaredCost) Gradient(actual, expected, gradient []float64) {
	for i, x := range expected {
		gradient[i] = actual[i] - x
	}
}
