package svm

// A GradientDescentSolver solves Problems using gradient descent.
//
// The algorithm that this uses has the following attributes:
// - This uses the Lagrangian dual problem, rather than the primal problem.
// - The gradients are computed using a closed-form mathematical function, not an approximation.
// - After each descent step, the coefficients are "projected" onto the constraints.
type GradientDescentSolver struct {
	// Steps indicates how many iterations the solver should perform.
	Steps int

	// Tradeoff specifies how important it is to minimize the magnitude of the normal vector versus
	// finding a good separation of samples.
	// In other words, it determines how important a wide margin is.
	// For linearly separable data, you should use a small (but non-zero) Tradeoff value.
	Tradeoff float64
}

func (c *GradientDescentSolver) Solve(p *Problem) *CombinationClassifier {
	s := newActiveSetSolver(p, 1/(2*c.Tradeoff))

	for i := 0; i < c.Steps; i++ {
		s.StepGradient()
	}

	return s.Solution(p)
}
