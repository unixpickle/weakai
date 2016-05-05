package svm

import (
	"time"

	"github.com/unixpickle/num-analysis/linalg"
)

// A GradientDescentSolver solves Problems using
// active set gradient descent.
type GradientDescentSolver struct {
	// Timeout specifies how long the algorithm
	// should run before returning the best
	// solution it's found.
	Timeout time.Duration

	// Tradeoff determines how important a wide
	// separation margin is. The greater the
	// tradeoff, the more the separation is
	// prioritized over the accuracy.
	//
	// For linearly separable data, a small
	// Tradeoff value will do.
	Tradeoff float64
}

func (c *GradientDescentSolver) Solve(p *Problem) *CombinationClassifier {
	iter := newGradientIterator(p, 1/(2*c.Tradeoff))
	timeout := time.After(c.Timeout)

	// TODO: this.

	/*
			lastValue := .QuadraticValue()

		StepLoop:
			for {
				s.StepGradient()
				newVal := s.QuadraticValue()
				if newVal >= lastValue && !s.ConstraintsChanged() {
					break
				}
				lastValue = newVal

				select {
				case <-timeout:
					break StepLoop
				default:
				}
			}

			return s.Solution(p)
	*/
}

type gradientIterator struct {
	activeSet *activeSet

	// This is A from the quadratic form
	// 1/2*c'*A*c - b'*c.
	matrix *linalg.Matrix

	// This is the current approximate solution,
	// or c from the above quadratic form.
	solution linalg.Vector

	stepCount          int
	constraintsChanged bool
}

func newGradientIterator(p *Problem, maxCoeff float64) *gradientIterator {
	varCount := len(p.Positives) + len(p.Negatives)
	posCount := len(p.Positives)
	signVec := make(linalg.Vector, varCount)

	for i := 0; i < len(p.Positives); i++ {
		signVec[i] = 1
	}
	for i := 0; i < len(p.Negatives); i++ {
		signVec[i+posCount] = -1
	}

	res := &gradientIterator{
		activeSet: newActiveSet(signVec, maxCoeff),
		matrix:    linalg.NewMatrix(varCount, varCount),
		solution:  make(linalg.Vector, varCount),
	}

	for i := 0; i < varCount; i++ {
		var iSample Sample
		if i >= posCount {
			iSample = p.Negatives[i-posCount]
		} else {
			iSample = p.Positives[i]
		}
		res.matrix.Set(i, i, p.Kernel(iSample, iSample))
		for j := 0; j < i; j++ {
			var jSample Sample
			if j >= posCount {
				jSample = p.Negatives[j-posCount]
			} else {
				jSample = p.Positives[j]
			}
			product := p.Kernel(iSample, jSample)
			res.matrix.Set(i, j, product)
			res.matrix.Set(j, i, product)
		}
	}

	for i := 0; i < varCount; i++ {
		for j := 0; j < varCount; j++ {
			oldVal := res.matrix.Get(i, j)
			oldVal *= signVec[i]
			oldVal *= signVec[j]
			res.matrix.Set(i, j, oldVal)
		}
	}

	return res
}
