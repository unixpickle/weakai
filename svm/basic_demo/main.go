package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/unixpickle/weakai/svm"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	positives := []svm.Sample{
		svm.Sample{V: []float64{0, 0}},
		svm.Sample{V: []float64{0, 0.3}},
		svm.Sample{V: []float64{0.3, 0}},
		svm.Sample{V: []float64{0.1, 0.1}},
	}

	negatives := []svm.Sample{
		svm.Sample{V: []float64{1, 1}},
		svm.Sample{V: []float64{1, 0.7}},
		svm.Sample{V: []float64{0.7, 1}},
		svm.Sample{V: []float64{0.9, 0.9}},
	}

	problem := &svm.Problem{
		Positives: positives,
		Negatives: negatives,
		Kernel:    svm.LinearKernel,
	}

	solution := svm.RandomlySolveLinear(problem, 100000, 5)
	fmt.Println("Solution from random solver:", solution)

	subgradientSolver := &svm.SubgradientSolver{
		Tradeoff: 0.001,
		Steps:    10000,
		StepSize: 0.001,
	}
	solution = subgradientSolver.Solve(problem)

	fmt.Println("Solution from subgradient solver:", solution)

	gradientSolver := &svm.GradientDescentSolver{
		Tradeoff: 0.001,
		StepSize: 0.001,
		Steps:    100000,
	}
	solution = gradientSolver.Solve(problem).Linearize()

	fmt.Println("Solution from gradient descent solver:", solution)
}
