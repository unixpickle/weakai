package svm

import (
	"math"
	"testing"
	"time"
)

func TestGradientSolverPolyKernel(t *testing.T) {
	positives := []Sample{
		{V: []float64{0.5, 0}},
		{V: []float64{0.4, 0.1}},
		{V: []float64{0.3, 0.2}},
		{V: []float64{0.5, 0.9}},
		{V: []float64{0.6, 1}},
		{V: []float64{0.4, 0.85}},
		{V: []float64{0.5, 0.5}},
	}

	negatives := []Sample{
		{V: []float64{0, 0.5}},
		{V: []float64{0.1, 0.4}},
		{V: []float64{0.9, 0.5}},
		{V: []float64{1, 0.6}},
		{V: []float64{0, 0.46}},
	}

	problem := &Problem{
		Positives: positives,
		Negatives: negatives,
		Kernel:    PolynomialKernel(1, 2),
	}

	gradientSolver := &GradientDescentSolver{
		Tradeoff: 0.0001,
		Timeout:  time.Minute,
	}
	solution := gradientSolver.Solve(problem)

	minPositive := math.Inf(1)
	maxNegative := math.Inf(-1)
	for _, x := range positives {
		minPositive = math.Min(minPositive, solution.Rating(x))
	}
	for _, x := range negatives {
		maxNegative = math.Max(maxNegative, solution.Rating(x))
	}
	if math.Abs(minPositive-1) > 1e-6 {
		t.Error("minPositive should be 1 but it's", minPositive)
	}
	if math.Abs(maxNegative+1) > 1e-6 {
		t.Error("maxNegative should be -1 but it's", maxNegative)
	}
}
