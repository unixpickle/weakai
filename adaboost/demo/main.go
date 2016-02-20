package main

import (
	"fmt"
	"math/rand"

	"github.com/unixpickle/weakai/adaboost"
)

const Dimensions = 2

func main() {
	problem := &adaboost.Problem{}
	for i := 0; i < 20; i++ {
		positive := (rand.Intn(2) == 1)
		sample := make([]float64, Dimensions)
		for d := 0; d < Dimensions; d++ {
			sample[d] = rand.Float64()
		}
		problem.Samples = append(problem.Samples, sample)
		problem.Classifications = append(problem.Classifications, positive)
	}
	stumps := adaboost.AllTreeStumps(problem.Samples, Dimensions)
	for _, s := range stumps {
		problem.Classifiers = append(problem.Classifiers, s)
	}

	solution := problem.Solve()
	for n := 1; n <= len(solution.Classifiers); n++ {
		partial := solution.PartialSolution(n)
		fmt.Println("After", n, "iterations, we get", errorCount(partial, problem), "wrong.")
	}
}

func errorCount(s *adaboost.Solution, p *adaboost.Problem) int {
	var count int
	for i, sam := range p.Samples {
		if s.Classify(sam) != p.Classifications[i] {
			count++
		}
	}
	return count
}
