package svm

import (
	"math"
	"math/rand"
)

// RandomlySolveLinear guesses random pre-images for hyperplane normals and returns the best guess,
// judging based on which guess yields the maximum separation between any two samples while
// maintaining an optimal margin size.
//
// This only works well for linear kernels.
// For nonlinear kernels, this may never guess the correct solution, since said solution will be in
// the transformed space and may not have a corresponding vector in the sample space.
func RandomlySolveLinear(p *Problem, numGuesses int, maxEntry float64) *LinearClassifier {
	var bestClassifier *LinearClassifier
	var bestTotalError float64
	var bestMagnitude float64

	for i := 0; i < numGuesses; i++ {
		guess := randomSample(len(p.Positives[0].V), maxEntry)
		mag := p.Kernel(guess, guess)
		threshold := idealThresholdForGuess(guess, p)

		totalError := 0.0
		for _, pos := range p.Positives {
			totalError += math.Max(0, 1-(p.Kernel(guess, pos)+threshold))
		}
		for _, neg := range p.Negatives {
			totalError += math.Max(0, 1+(p.Kernel(guess, neg)+threshold))
		}

		if i == 0 || (totalError == bestTotalError && mag < bestMagnitude) ||
			totalError < bestTotalError {
			bestTotalError = totalError
			bestMagnitude = mag
			bestClassifier = &LinearClassifier{
				HyperplaneNormal: guess,
				Threshold:        threshold,
				Kernel:           p.Kernel,
			}
		}
	}
	return bestClassifier
}

func randomSample(dimension int, componentMax float64) Sample {
	vec := make([]float64, dimension)
	for i := range vec {
		vec[i] = (rand.Float64() - 0.5) * componentMax * 2
	}
	return Sample{V: vec}
}

func idealThresholdForGuess(guess Sample, p *Problem) float64 {
	var minPositiveDot float64
	for i, positive := range p.Positives {
		product := p.Kernel(guess, positive)
		if i == 0 || product < minPositiveDot {
			minPositiveDot = product
		}
	}

	var maxNegativeDot float64
	for i, negative := range p.Negatives {
		product := p.Kernel(guess, negative)
		if i == 0 || product > maxNegativeDot {
			maxNegativeDot = product
		}
	}

	return -(minPositiveDot + maxNegativeDot) / 2
}
