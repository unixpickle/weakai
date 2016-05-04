package svm

import "time"

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
	s := newActiveSetSolver(p, 1/(2*c.Tradeoff))
	timeout := time.After(c.Timeout)

	lastValue := s.QuadraticValue()

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
}
