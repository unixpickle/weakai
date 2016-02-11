package svm

import (
	"math"
	"math/rand"
)

// RandomlySolve generates many random guesses for solutions and returns the "best" guess,
// judging based on which guess yields the maximum separation between any two samples.
func RandomlySolve(p *Problem, numGuesses int, maxWeight float64) *LinearClassifier {
	var bestClassifier *LinearClassifier
	var bestSeparation float64

	sampleCount := len(p.Negatives) + len(p.Positives)
	for i := 0; i < numGuesses; i++ {
		guess := randomSample(sampleCount, maxWeight)
		normalVector := make(Sample, len(p.Positives[0]))
		for i, positive := range p.Positives {
			coefficient := guess[i]
			for j, x := range normalVector {
				normalVector[j] = x + coefficient*positive[j]
			}
		}
		for i, negative := range p.Negatives {
			coefficient := guess[i+len(p.Positives)]
			for j, x := range normalVector {
				normalVector[j] = x + coefficient*negative[j]
			}
		}

		var minPositiveDot float64
		var closestPositive Sample
		for i, positive := range p.Positives {
			product := p.Kernel(normalVector, positive)
			if i == 0 || product < minPositiveDot {
				closestPositive = positive
				minPositiveDot = product
			}
		}

		var maxNegativeDot float64
		var closestNegative Sample
		for i, negative := range p.Negatives {
			product := p.Kernel(normalVector, negative)
			if i == 0 || product > maxNegativeDot {
				closestNegative = negative
				maxNegativeDot = product
			}
		}

		separation := (p.Kernel(closestPositive, normalVector) -
			p.Kernel(closestNegative, normalVector)) /
			math.Sqrt(p.Kernel(normalVector, normalVector))

		if separation > bestSeparation || i == 0 {
			bestSeparation = separation
			bestClassifier = &LinearClassifier{
				HyperplaneNormal: normalVector,
				Threshold:        -(minPositiveDot + maxNegativeDot) / 2,
				Kernel:           p.Kernel,
			}
		}
	}
	return bestClassifier
}

func randomSample(dimension int, componentMax float64) Sample {
	res := make(Sample, dimension)
	for i := range res {
		res[i] = (rand.Float64() - 0.5) * componentMax * 2
	}
	return res
}
