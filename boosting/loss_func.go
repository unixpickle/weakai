package boosting

import (
	"math"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A LossFunc is a differentiable function which can
// measure the difference between two classification
// vectors.
type LossFunc interface {
	// Loss evaluates the loss function to find the
	// distance between the actual classification and
	// the expected (desired) classification.
	// The resulting autofunc.Result should contain
	// one component: a scaler measure of the error.
	Loss(actual autofunc.Result, expected linalg.Vector) autofunc.Result

	// OptimalStep returns the value a which minimizes
	// Loss(x0+a*x, y).
	// In other words, it performs a line search to
	// find the optimal step size.
	OptimalStep(x0, x, y linalg.Vector) float64
}

// SquareLoss is a loss function which squares the
// distances between desired and actual classifications.
type SquareLoss struct{}

// Loss returns the squared magnitude of the difference
// between actual and expected.
func (_ SquareLoss) Loss(actual autofunc.Result, expected linalg.Vector) autofunc.Result {
	expVar := &autofunc.Variable{Vector: expected.Copy().Scale(-1)}
	return autofunc.SumAll(autofunc.Square(autofunc.Add(actual, expVar)))
}

// OptimalStep returns the value a which minimizes
// the squared distance between y and x0+a*x.
func (_ SquareLoss) OptimalStep(x0, x, y linalg.Vector) float64 {
	return (x.Dot(y) - x.Dot(x0)) / x.Dot(x)
}

// ExpLoss is the exponential loss function used in
// the AdaBoost algorithm.
type ExpLoss struct{}

// Loss returns the exponential loss, as given by
// exp(-actual*expected).
func (_ ExpLoss) Loss(actual autofunc.Result, expected linalg.Vector) autofunc.Result {
	expVar := &autofunc.Variable{Vector: expected.Copy().Scale(-1)}
	dots := autofunc.Mul(actual, expVar)
	exps := autofunc.Exp{}.Apply(dots)
	return autofunc.SumAll(exps)
}

// OptimalStep returns the value which minimizes
// the exponential loss between y and x0+a*x.
// It only works if all the entries of x and y are
// 1 or -1, although the entries of x0 are not
// restricted in any way.
func (_ ExpLoss) OptimalStep(x0, x, y linalg.Vector) float64 {
	var weightedRight, weightedWrong float64
	for i, xVal := range x {
		if xVal != -1 && xVal != 1 {
			panic("all entries of x must be 1 or -1")
		}
		yVal := y[i]
		if yVal != -1 && yVal != 1 {
			panic("all entries of y must be 1 or -1")
		}
		weight := math.Exp(-x0[i] * yVal)
		if xVal == yVal {
			weightedRight += weight
		} else {
			weightedWrong += weight
		}
	}
	return 0.5 * math.Log(weightedRight/weightedWrong)
}
