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
		{V: []float64{0.5, 0}},
		{V: []float64{0.4, 0.1}},
		{V: []float64{0.3, 0.2}},
		{V: []float64{0.5, 0.9}},
		{V: []float64{0.6, 1}},
		{V: []float64{0.4, 0.85}},
		{V: []float64{0.5, 0.5}},
	}

	negatives := []svm.Sample{
		{V: []float64{0, 0.5}},
		{V: []float64{0.1, 0.4}},
		{V: []float64{0.9, 0.5}},
		{V: []float64{1, 0.6}},
		{V: []float64{0, 0.46}},
	}

	problem := &svm.Problem{
		Positives: positives,
		Negatives: negatives,
		Kernel:    svm.PolynomialKernel(1, 2),
	}

	printProblem(problem)

	printSolution("random linear", svm.RandomlySolveLinear(problem, 100000, 5), problem)

	subgradientSolver := &svm.SubgradientSolver{
		Tradeoff: 0.001,
		StepSize: 0.1,
		Steps:    100000,
	}
	printSolution("subgradient descent", subgradientSolver.Solve(problem), problem)

	gradientSolver := &svm.GradientDescentSolver{
		Tradeoff: 0.001,
		StepSize: 0.0001,
		Steps:    500000,
	}
	printSolution("gradient descent", gradientSolver.Solve(problem), problem)
}

func printProblem(p *svm.Problem) {
	grid := make([]rune, 10*20)
	x := func(samples []svm.Sample, char rune) {
		for _, s := range samples {
			x, y := int(s.V[0]*20), int(s.V[1]*10)
			if x >= 20 {
				x = 19
			}
			if y >= 10 {
				y = 9
			}
			grid[x+y*20] = char
		}
	}
	x(p.Positives, '+')
	x(p.Negatives, '-')

	fmt.Println("Problem:")
	fmt.Println("----------------------")
	for y := 0; y < 10; y++ {
		fmt.Print("|")
		for x := 0; x < 20; x++ {
			if ch := grid[x+y*20]; ch == 0 {
				fmt.Print(" ")
			} else {
				fmt.Print(string(ch))
			}
		}
		fmt.Println("|")
	}
	fmt.Println("----------------------")
}

func printSolution(solverType string, solution svm.Classifier, p *svm.Problem) {
	fmt.Println("Got solution from", solverType)

	incorrectCount := 0
	for _, pos := range p.Positives {
		if !solution.Classify(pos) {
			incorrectCount++
		}
	}
	for _, neg := range p.Negatives {
		if solution.Classify(neg) {
			incorrectCount++
		}
	}

	fmt.Println("Number of incorrect classifications:", incorrectCount)

	fmt.Println("Visual representation:")
	for y := 0.0; y <= 1.0; y += 0.1 {
		fmt.Print(" ")
		for x := 0.0; x < 1.0; x += 0.05 {
			classification := solution.Classify(svm.Sample{V: []float64{x, y}})
			if classification {
				fmt.Print("+")
			} else {
				fmt.Print("-")
			}
		}
		fmt.Println("")
	}
	fmt.Println("")
}
