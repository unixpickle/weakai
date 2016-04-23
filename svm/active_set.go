package svm

import (
	"math"

	"github.com/unixpickle/num-analysis/kahan"
	"github.com/unixpickle/num-analysis/linalg"
)

// activeSetSolver uses an active set method
// to solve an SVM-style quadratic
// optimization problem.
type activeSetSolver struct {
	// This is A from the quadratic form
	// 1/2*c'*A*c - b'*c.
	a *linalg.Matrix

	// signVec is a vector of 1's or 1's
	// indicating whether each sample is
	// positive or negative.
	signVec linalg.Vector

	// maxCoeff is the maximum value any
	// component of c can have.
	maxCoeff float64

	// c is the current approximate solution.
	c linalg.Vector
}

func newActiveSetSolver(p *Problem, maxCoeff float64) *activeSetSolver {
	varCount := len(p.Positives) + len(p.Negatives)
	res := &activeSetSolver{
		a:        linalg.NewMatrix(varCount, varCount),
		signVec:  make(linalg.Vector, varCount),
		maxCoeff: maxCoeff,
		c:        make(linalg.Vector, varCount),
	}

	posCount := len(p.Positives)

	for i := 0; i < varCount; i++ {
		if i >= posCount {
			res.signVec[i] = -1
		} else {
			res.signVec[i] = 1
		}
	}

	for i := 0; i < varCount; i++ {
		var iSample Sample
		if i >= posCount {
			iSample = p.Negatives[i-posCount]
		} else {
			iSample = p.Positives[i]
		}
		res.a.Set(i, i, p.Kernel(iSample, iSample))
		for j := 0; j < i; j++ {
			var jSample Sample
			if j >= posCount {
				jSample = p.Negatives[j-posCount]
			} else {
				jSample = p.Positives[j]
			}
			product := p.Kernel(iSample, jSample)
			res.a.Set(i, j, product)
			res.a.Set(j, i, product)
		}
	}

	for i := 0; i < varCount; i++ {
		for j := 0; j < varCount; j++ {
			oldVal := res.a.Get(i, j)
			oldVal *= res.signVec[i]
			oldVal *= res.signVec[j]
			res.a.Set(i, j, oldVal)
		}
	}

	return res
}

// StepGradient performs a step of constrained
// gradient descent.
func (s *activeSetSolver) StepGradient() {
	stepDirection := s.gradient().Scale(-1)

	projectedSignVec := s.signVec.Copy()
	for i, x := range stepDirection {
		if x > 0 && s.c[i] >= s.maxCoeff {
			stepDirection[i] = 0
			projectedSignVec[i] = 0
		} else if x < 0 && s.c[i] <= 0 {
			stepDirection[i] = 0
			projectedSignVec[i] = 0
		}
	}

	projAmount := projectedSignVec.Dot(stepDirection) / stepDirection.Dot(stepDirection)
	stepDirection.Add(projectedSignVec.Scale(-projAmount))

	s.step(stepDirection)
}

// Solution returns the current approximation of
// the solution.
func (s *activeSetSolver) Solution(p *Problem) *CombinationClassifier {
	solution := s.c.Copy()

	// TODO: delete support vectors with 0 coefficients.
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

func (s *activeSetSolver) gradient() linalg.Vector {
	columnMat := &linalg.Matrix{
		Rows: s.a.Cols,
		Cols: 1,
		Data: s.c,
	}
	residual := s.a.Mul(columnMat)
	for i, x := range residual.Data {
		residual.Data[i] = x - 1
	}
	return linalg.Vector(residual.Data)
}

func (s *activeSetSolver) step(d linalg.Vector) {
	// Avoid stepping so much that we break
	// inactive inequality constraints.
	var maxStep, minStep float64
	var maxIdx, minIdx int
	isFirst := true
	for i, x := range d {
		if x == 0 {
			continue
		}
		guessVal := s.c[i]
		maxValue := (s.maxCoeff - guessVal) / x
		minValue := -guessVal / x
		if x < 0 {
			maxValue, minValue = minValue, maxValue
		}
		if isFirst {
			isFirst = false
			minStep, maxStep = minValue, maxValue
			maxIdx, minIdx = i, i
		} else {
			if minValue > minStep {
				minStep = minValue
				minIdx = i
			}
			if maxValue < maxStep {
				maxStep = maxValue
				maxIdx = i
			}
		}
	}

	constrainedIdx := -1

	unconstrainedStep := s.optimalStep(d)
	if unconstrainedStep < minStep {
		s.c = d.Scale(minStep).Add(s.c)
		constrainedIdx = minIdx
	} else if unconstrainedStep > maxStep {
		s.c = d.Scale(maxStep).Add(s.c)
		constrainedIdx = maxIdx
	} else {
		s.c = d.Scale(unconstrainedStep).Add(s.c)
	}

	if constrainedIdx >= 0 {
		val := s.c[constrainedIdx]
		if math.Abs(val) > math.Abs(val-s.maxCoeff) {
			s.c[constrainedIdx] = s.maxCoeff
		} else {
			s.c[constrainedIdx] = 0
		}
	}
}

func (s *activeSetSolver) optimalStep(d linalg.Vector) float64 {
	// The optimal step size is (d'*b - d'*A*x)/(d'*A*d)
	// where d is the direction, A is the matrix, x is
	// the current approximate solution, and b is all 1's.

	dMat := &linalg.Matrix{
		Rows: len(d),
		Cols: 1,
		Data: d,
	}
	ad := linalg.Vector(s.a.Mul(dMat).Data)

	summer := kahan.NewSummer64()
	for _, x := range d {
		summer.Add(x)
	}

	numerator := summer.Sum() - s.c.Dot(ad)
	denominator := d.Dot(ad)

	return numerator / denominator
}
