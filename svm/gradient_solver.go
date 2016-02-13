package svm

import "math"

// A GradientDescentSolver solves Problems using gradient descent.
//
// The algorithm that this uses has the following attributes:
// - This uses the Lagrangian dual problem, rather than the primal problem.
// - The gradients are computed using a closed-form mathematical function, not an approximation.
// - After each descent step, the coefficients are "projected" onto the constraints.
type GradientDescentSolver struct {
	// Steps indicates how many iterations the solver should perform.
	Steps int

	// StepSize is a number between 0 and 1 which determines how much of the gradient should be
	// added to the current solution at each step.
	// There is a tradeoff between a small step size and a small number of steps.
	StepSize float64

	// Tradeoff specifies how important it is to minimize the magnitude of the normal vector versus
	// finding a good separation of samples.
	// In other words, it determines how important a wide margin is.
	// For linearly separable data, you should use a small (but non-zero) Tradeoff value.
	Tradeoff float64
}

func (c *GradientDescentSolver) Solve(p *Problem) *CombinationClassifier {
	solution := make([]float64, len(p.Positives)+len(p.Negatives))
	temp := make([]float64, len(solution))

	for i := 0; i < c.Steps; i++ {
		for j := range solution {
			temp[j] = c.partial(p, solution, j)
		}
		for j, partial := range temp {
			solution[j] += partial * c.StepSize
		}
		c.constraintProjection(p, solution)
	}

	// TODO: figure out the support vectors by weeding out samples which have ~0 coefficients.
	supportVectors := make([]Sample, len(p.Positives)+len(p.Negatives))
	copy(supportVectors, p.Positives)
	copy(supportVectors[len(p.Positives):], p.Negatives)
	for i := len(p.Positives); i < len(p.Positives)+len(p.Negatives); i++ {
		solution[i] *= -1
	}

	res := &CombinationClassifier{
		SupportVectors: supportVectors,
		Coefficients:   solution,
		Kernel:         p.Kernel,
	}
	res.computeThreshold(p)
	return res
}

// partial computes the partial derivative of the dual optimization problem with respect to one of
// the coefficients, given by the coefficient index.
func (c *GradientDescentSolver) partial(p *Problem, coefficients []float64, idx int) float64 {
	var coefficientSample Sample
	sampleCoefficient := 1.0
	if idx >= len(p.Positives) {
		coefficientSample = p.Negatives[idx-len(p.Positives)]
		sampleCoefficient = -1
	} else {
		coefficientSample = p.Positives[idx]
	}

	// I got this by differentiating sum(ci) - 1/2*sum(sum(yi*ci*yj*cj*(xi*xj))) with respect to ci.
	partial := 1.0
	for i, positive := range p.Positives {
		product := p.Kernel(coefficientSample, positive)
		partial -= sampleCoefficient * coefficients[i] * product
	}
	for i, negative := range p.Negatives {
		product := p.Kernel(coefficientSample, negative)
		partial += sampleCoefficient * coefficients[i+len(p.Positives)] * product
	}
	return partial
}

// constraintProjection projects the current solution onto two constraints:
// - sum(ci+yi)=0, where yi is the sign of a given sample, and ci is its coefficient.
// - 0 <= ci <= 1/k where ci is a coefficient and k is the width-separation tradeoff.
func (c *GradientDescentSolver) constraintProjection(p *Problem, coefficients []float64) {
	// Let the vector y = (y1, y2, ..., yn) where yi is -1 for negatives and 1 for positives.
	// Let the vector c = (c1, c2, ..., cn) where ci is the i-th coefficient.
	yDotY := float64(len(p.Positives) + len(p.Negatives))
	var yDotC float64
	for i, coeff := range coefficients {
		if i < len(p.Positives) {
			yDotC += coeff
		} else {
			yDotC -= coeff
		}
	}

	// We now know y*c and ||y||^2, so c projected onto y is y*((y*c)/||y||^2), and thus the
	// projection onto the space orthogonal to y is c - y*((y*c)/||y||^2)
	scaler := yDotC / yDotY
	for i := range coefficients {
		if i < len(p.Positives) {
			coefficients[i] -= scaler
		} else {
			coefficients[i] += scaler
		}
	}

	coefficientMax := 1 / c.Tradeoff
	for i, coeff := range coefficients {
		coefficients[i] = math.Min(coefficientMax, math.Max(0, coeff))
	}
}
