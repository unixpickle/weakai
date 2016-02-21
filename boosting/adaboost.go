package boosting

import "math"

type AdaboostSolver struct {
	MaxReuse int
}

func (a AdaboostSolver) Solve(p *Problem) *Solution {
	sol := &Solution{
		Classifiers: make([]Classifier, 0, len(p.Classifiers)),
		Weights:     make([]float64, 0, len(p.Classifiers)),
	}

	for iteration := 0; iteration < a.MaxReuse || iteration < 1; iteration++ {
		usedClassifiers := map[Classifier]bool{}
		for i := 0; i < len(p.Classifiers) && p.numCorrect(sol) < len(p.Samples); i++ {
			errors := a.errorsForSamples(p, sol)
			var bestErrorWeightSum float64
			var bestClassifier Classifier
			for _, c := range p.Classifiers {
				if usedClassifiers[c] {
					continue
				}
				var errorWeightSum float64
				for i, sample := range p.Samples {
					if c.Classify(sample) != p.Classifications[i] {
						errorWeightSum += errors[i]
					}
				}
				if errorWeightSum < bestErrorWeightSum || bestClassifier == nil {
					bestErrorWeightSum = errorWeightSum
					bestClassifier = c
				}
			}
			errorRate := bestErrorWeightSum / sumAll(errors)
			weight := 0.5 * math.Log((1-errorRate)/errorRate)
			usedClassifiers[bestClassifier] = true
			if iteration > 0 {
				for i, c := range sol.Classifiers {
					if c == bestClassifier {
						sol.Weights[i] += weight
					}
				}
			} else {
				sol.Weights = append(sol.Weights, weight)
				sol.Classifiers = append(sol.Classifiers, bestClassifier)
			}
		}
	}

	return sol
}

func (_ AdaboostSolver) errorsForSamples(p *Problem, s *Solution) []float64 {
	res := make([]float64, len(p.Samples))
	for i, sample := range p.Samples {
		classification := s.Evaluate(sample)
		coefficient := -1.0
		if !p.Classifications[i] {
			coefficient = 1.0
		}
		res[i] = math.Exp(coefficient * classification)
	}
	return res
}

func sumAll(vals []float64) float64 {
	var sum float64
	for _, v := range vals {
		sum += v
	}
	return sum
}
