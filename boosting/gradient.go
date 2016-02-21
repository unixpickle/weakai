package boosting

import "math/rand"

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
			sol.Weights[i] = rand.Float64()*float64(attempt) - float64(attempt)/2
		}

		partials := make([]float64, len(p.Classifiers))
		for i := 0; i < g.Iterations; i++ {
			errs := AdaboostSolver{}.errorsForSamples(p, sol)
			for i := range partials {
				partials[i] = g.partial(errs, sol, p, i)
			}
			for i, x := range partials {
				sol.Weights[i] -= x * g.StepSize
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
	sum := sumAll(f)
	for i, x := range f {
		f[i] = x / sum
	}
}
