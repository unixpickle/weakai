package svm

import (
	"math"
	"time"

	"github.com/unixpickle/num-analysis/kahan"
	"github.com/unixpickle/num-analysis/linalg"
)

const reprojectIterationCount = 100
const minProjectionMagChange = 1e-11

// A GradientDescentSolver solves Problems using
// active set gradient descent.
type GradientDescentSolver struct {
	// Timeout specifies how long the algorithm
	// should run before returning the best
	// solution it's found.
	// If this is zero, no timeout is used.
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
	sampleCount := float64(len(p.Positives) + len(p.Negatives))
	maxCoefficient := 1 / (2 * c.Tradeoff * sampleCount)
	iter := newGradientIterator(p, maxCoefficient)

	var timeout <-chan time.Time
	if c.Timeout != 0 {
		timeout = time.After(c.Timeout)
	}

	lastValue := iter.QuadraticValue()

StepLoop:
	for {
		iter.StepGradient()
		newVal := iter.QuadraticValue()
		if iter.ShouldTerminate() || (newVal >= lastValue && !iter.ConstraintsChanged()) {
			break
		}
		lastValue = newVal

		if timeout != nil {
			select {
			case <-timeout:
				break StepLoop
			default:
			}
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
	shouldTerminate    bool

	gradientCache  linalg.Vector
	quadraticCache float64
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

	res.updateCaches()

	return res
}

// StepGradient performs a step of constrained
// gradient descent.
func (g *gradientIterator) StepGradient() {
	stepDirection := g.gradientCache.Scale(-1)
	g.gradientCache = nil

	g.constraintsChanged = g.activeSet.Prune(stepDirection)
	preAbs := stepDirection.MaxAbs()
	g.activeSet.ProjectOut(stepDirection)
	postAbs := stepDirection.MaxAbs()
	if postAbs < preAbs*minProjectionMagChange {
		g.shouldTerminate = true
		return
	}

	g.step(stepDirection)
	g.updateCaches()
}

// QuadraticValue returns the current value of
// the minimizing quadratic form.
func (g *gradientIterator) QuadraticValue() float64 {
	return g.quadraticCache
}

// ConstraintsChanged returns whether or not
// the last step changed any constraints.
func (g *gradientIterator) ConstraintsChanged() bool {
	return g.constraintsChanged
}

// ShouldTerminate returns true if some
// obvious condition (e.g. zero gradient)
// was met to terminate the iterations.
func (g *gradientIterator) ShouldTerminate() bool {
	return g.shouldTerminate
}

// Solution returns the current approximation of
// the solution.
func (g *gradientIterator) Solution(p *Problem) *CombinationClassifier {
	supportVectors := make([]Sample, 0, len(p.Positives)+len(p.Negatives))
	solution := make([]float64, 0, len(p.Positives)+len(p.Negatives))
	for i, x := range g.solution {
		if x != 0 {
			if i < len(p.Positives) {
				supportVectors = append(supportVectors, p.Positives[i])
			} else {
				neg := p.Negatives[i-len(p.Positives)]
				supportVectors = append(supportVectors, neg)
				x *= -1
			}
			solution = append(solution, x)
		}
	}

	res := &CombinationClassifier{
		SupportVectors: supportVectors,
		Coefficients:   solution,
		Kernel:         p.Kernel,
	}
	res.computeThreshold(p)

	return res
}

func (g *gradientIterator) step(d linalg.Vector) {
	optimalAmount := g.optimalStep(d)
	if optimalAmount < 0 {
		g.shouldTerminate = true
		return
	}
	if g.activeSet.Step(g.solution, d, optimalAmount) {
		g.constraintsChanged = true
	}

	g.stepCount++
	if g.stepCount%reprojectIterationCount == 0 {
		g.reprojectConstraints()
		g.constraintsChanged = true
	}
}

func (g *gradientIterator) updateCaches() {
	// This function computes and uses various pieces
	// of the quadratic form 1/2*c'*A*c - b'*c.

	columnMat := &linalg.Matrix{
		Rows: g.matrix.Cols,
		Cols: 1,
		Data: g.solution,
	}

	// Compute A*c.
	columnProduct := g.matrix.Mul(columnMat)

	// Compute c'*A*c.
	quadValue := kahan.NewSummer64()
	for i, x := range g.solution {
		quadValue.Add(x * columnProduct.Data[i])
	}

	// Compute b'*c
	linearValue := kahan.NewSummer64()
	for _, x := range g.solution {
		linearValue.Add(x)
	}

	g.quadraticCache = quadValue.Sum()*0.5 - linearValue.Sum()

	// Compute A*c - b.
	for i, x := range columnProduct.Data {
		columnProduct.Data[i] = x - 1
	}
	g.gradientCache = columnProduct.Data
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

func (g *gradientIterator) reprojectConstraints() {
	// This is a crude way to ensure that the current
	// solution is a feasible point. It would give
	// horrible results if the solution were not already
	// close to being feasible, but since it is only used
	// to fix small rounding errors, it should work fine.

	for i, x := range g.solution {
		g.solution[i] = math.Max(0, math.Min(g.activeSet.MaxCoeff, x))
	}

	signVec := g.activeSet.SignVec
	currentDot := g.solution.Dot(signVec)

	changed := true
	for changed {
		changed = false
		for i, x := range g.solution {
			desiredChange := -signVec[i] * currentDot
			if x+desiredChange < 0 {
				if x > 0 {
					g.solution[i] = 0
					currentDot = g.solution.Dot(signVec)
					changed = true
				}
			} else if x+desiredChange > g.activeSet.MaxCoeff {
				if x < g.activeSet.MaxCoeff {
					g.solution[i] = g.activeSet.MaxCoeff
					currentDot = g.solution.Dot(signVec)
					changed = true
				}
			} else {
				g.solution[i] += desiredChange
				return
			}
		}
	}
}
