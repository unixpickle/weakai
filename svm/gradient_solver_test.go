package svm

import (
	"math"
	"testing"
	"time"

	"github.com/unixpickle/num-analysis/linalg"
)

const benchmarkDimensionality = 100

func TestGradientSolverLinear(t *testing.T) {
	problem, supportVec := linearSVMProblem(20)
	solver := &GradientDescentSolver{
		Tradeoff: 0.0001,
		Timeout:  time.Minute,
	}
	solution := solver.Solve(problem)

	if math.Abs(solution.Threshold) > 1e-5 {
		t.Error("unexpected threshold:", solution.Threshold)
	}

	if len(solution.SupportVectors) != 38 {
		t.Error("unexpected number of support vectors:", len(solution.SupportVectors))
	}

	normal := solution.Linearize().HyperplaneNormal.V
	for i, x := range supportVec {
		if math.Abs(x-normal[i]) > 1e-5 {
			t.Fatal("unexpected support vector:", normal)
			return
		}
	}
}

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

func BenchmarkGradientSolver(b *testing.B) {
	problem, _ := linearSVMProblem(benchmarkDimensionality)

	solver := &GradientDescentSolver{
		Tradeoff: 0.0001,
		Timeout:  time.Hour,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		solver.Solve(problem)
	}
}

func linearSVMProblem(size int) (*Problem, linalg.Vector) {
	supportVector := make(linalg.Vector, size)
	for i := range supportVector {
		supportVector[i] = float64(i%5) - 2.5
	}
	supportVector.Scale(1.0 / math.Sqrt(supportVector.Dot(supportVector)))
	doubleSupport := supportVector.Copy().Scale(2)

	negVector := supportVector.Copy().Scale(-1)
	doubleNeg := supportVector.Copy().Scale(-2)

	positives := make([]Sample, (size-1)*2)
	negatives := make([]Sample, len(positives))

	for i := 0; i < size-1; i++ {
		orthoVector := make(linalg.Vector, len(supportVector))
		orthoVector[i+1] = 1
		orthoVector[0] = -supportVector[i+1] / supportVector[0]
		positives[i*2] = Sample{V: []float64(orthoVector.Copy().Add(supportVector))}
		positives[i*2+1] = Sample{V: []float64(orthoVector.Copy().Add(doubleSupport))}
		negatives[i*2] = Sample{V: []float64(orthoVector.Copy().Add(negVector))}
		negatives[i*2+1] = Sample{V: []float64(orthoVector.Copy().Add(doubleNeg))}
	}

	problem := &Problem{
		Positives: positives,
		Negatives: negatives,
		Kernel:    LinearKernel,
	}

	return problem, supportVector
}
