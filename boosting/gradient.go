package boosting

import (
	"math"
	"math/rand"
)

type GradientSolver struct {
	Attempts   int
	Iterations int
	StepSize   float64
}

func (g *GradientSolver) Solve(p *Problem) *Solution {
	var res *Solution
	var numRight int

	for attempt := 0; attempt < g.Attempts || attempt < 1; attempt++ {
		sol := &Solution{
			Classifiers: p.Classifiers,
			Weights:     make([]float64, len(p.Classifiers)),
		}
		for i := range p.Classifiers {
			sol.Weights[i] = rand.Float64() - 0.5
		}

		partials := make([]float64, len(p.Classifiers))
		for i := 0; i < g.Iterations; i++ {
			errs := AdaboostSolver{}.errorsForSamples(p, sol)
			var gradientMag float64
			for i := range partials {
				partials[i] = g.partial(errs, sol, p, i)
				gradientMag += math.Pow(partials[i], 2)
			}
			gradientMag = math.Sqrt(gradientMag)
			for i, x := range partials {
				sol.Weights[i] -= x * g.StepSize / gradientMag
			}
			normalizeWeights(sol.Weights)
		}

		right := p.numCorrect(sol)
		if right > numRight || attempt == 0 {
			numRight = right
			res = sol
		}
	}

	return res
}

func (g *GradientSolver) partial(errs []float64, s *Solution, p *Problem, index int) float64 {
	var partial float64
	for i, sample := range p.Samples {
		sign := -1.0
		if s.Classifiers[index].Classify(sample) != p.Classifications[i] {
			sign = 1.0
		}
		partial += sign * errs[i]
	}
	return partial
}

func normalizeWeights(f []float64) {
	var sum float64
	for _, x := range f {
		sum += math.Pow(x, 2)
	}
	sum = math.Sqrt(sum)
	for i, x := range f {
		f[i] = x / sum
	}
}
