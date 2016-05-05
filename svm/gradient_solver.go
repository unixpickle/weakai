package svm

import (
	"math"
	"time"

	"github.com/unixpickle/num-analysis/kahan"
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

	lastValue := iter.QuadraticValue()

StepLoop:
	for {
		iter.StepGradient()
		newVal := iter.QuadraticValue()
		if newVal >= lastValue && !iter.ConstraintsChanged() {
			break
		}
		lastValue = newVal

		select {
		case <-timeout:
			break StepLoop
		default:
		}
	}

	return iter.Solution(p)
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

// StepGradient performs a step of constrained
// gradient descent.
func (g *gradientIterator) StepGradient() {
	g.constraintsChanged = false
	stepDirection := g.gradient().Scale(-1)
	g.constraintsChanged = g.activeSet.Prune(stepDirection)
	g.activeSet.ProjectOut(stepDirection)
	g.step(stepDirection)
}

// QuadraticValue returns the current value of
// the minimizing quadratic form.
func (g *gradientIterator) QuadraticValue() float64 {
	columnMat := &linalg.Matrix{
		Rows: g.matrix.Cols,
		Cols: 1,
		Data: g.solution,
	}
	result := kahan.NewSummer64()
	quadTerm := columnMat.Transpose().Mul(g.matrix).Mul(columnMat).Data[0]
	result.Add(quadTerm * 0.5)
	for _, x := range g.solution {
		result.Add(-x)
	}
	return result.Sum()
}

// ConstraintsChanged returns whether or not
// the last step changed any constraints.
func (g *gradientIterator) ConstraintsChanged() bool {
	return g.constraintsChanged
}

// Solution returns the current approximation of
// the solution.
func (g *gradientIterator) Solution(p *Problem) *CombinationClassifier {
	solution := g.solution.Copy()

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

func (g *gradientIterator) gradient() linalg.Vector {
	columnMat := &linalg.Matrix{
		Rows: g.matrix.Cols,
		Cols: 1,
		Data: g.solution,
	}
	residual := g.matrix.Mul(columnMat)
	for i, x := range residual.Data {
		residual.Data[i] = x - 1
	}
	return linalg.Vector(residual.Data)
}

func (g *gradientIterator) optimalStep(d linalg.Vector) float64 {
	// The optimal step size is (d'*b - c'*A*d)/(d'*A*d)
	// where d is the direction, A is the matrix, x is
	// the current approximate solution, and b is all 1's.

	dMat := &linalg.Matrix{
		Rows: len(d),
		Cols: 1,
		Data: d,
	}
	ad := linalg.Vector(g.matrix.Mul(dMat).Data)

	summer := kahan.NewSummer64()
	for _, x := range d {
		summer.Add(x)
	}

	numerator := summer.Sum() - g.solution.Dot(ad)
	denominator := d.Dot(ad)

	return numerator / denominator
}

func (g *gradientIterator) step(d linalg.Vector) {
	optimalAmount := g.optimalStep(d)
	if g.activeSet.Step(g.solution, d, optimalAmount) {
		g.constraintsChanged = true
	}

	g.stepCount++
	if g.stepCount%reprojectIterationCount == 0 {
		g.reprojectConstraints()
		g.constraintsChanged = true
	}
}

func (g *gradientIterator) reprojectConstraints() {
	for i, x := range g.solution {
		g.solution[i] = math.Max(0, math.Min(g.activeSet.MaxCoeff, x))
	}
	signVec := g.activeSet.SignVec
	projAmount := signVec.Dot(g.solution) / signVec.Dot(signVec)
	g.solution.Add(signVec.Copy().Scale(-projAmount))
}
