package neuralnet

import (
	"sync"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A CostFunc is a cost function (aka loss function)
// used to train a neural network.
//
// It may be beneficial for CostFuncs to lazily
// compute their outputs, since they may be used
// solely for their derivatives.
type CostFunc interface {
	Cost(expected linalg.Vector, actual autofunc.Result) autofunc.Result
	CostR(v autofunc.RVector, expected linalg.Vector,
		actual autofunc.RResult) autofunc.RResult
}

// TotalCost returns the total cost of a layer on a
// set of VectorSamples.
// The elements of s must be VectorSamples.
func TotalCost(c CostFunc, layer autofunc.Func, s SampleSet) float64 {
	var totalCost float64
	for _, sample := range s {
		vs := sample.(VectorSample)
		inVar := &autofunc.Variable{vs.Input}
		result := layer.Apply(inVar)
		costOut := c.Cost(vs.Output, result)
		totalCost += costOut.Output()[0]
	}
	return totalCost
}

// MeanSquaredCost computes the cost as ||a-x||^2
// where a is the actual output and x is the desired
// output.
type MeanSquaredCost struct{}

func (_ MeanSquaredCost) Cost(x linalg.Vector, a autofunc.Result) autofunc.Result {
	return &meanSquaredResult{
		Actual:   a,
		Expected: x,
	}
}

func (_ MeanSquaredCost) CostR(v autofunc.RVector, a linalg.Vector,
	x autofunc.RResult) autofunc.RResult {
	aVar := &autofunc.Variable{a.Copy().Scale(-1)}
	aVarR := autofunc.NewRVariable(aVar, v)
	return autofunc.SquaredNorm{}.ApplyR(v, autofunc.AddR(aVarR, x))
}

type meanSquaredResult struct {
	OutputLock   sync.RWMutex
	OutputVector linalg.Vector

	Actual   autofunc.Result
	Expected linalg.Vector
}

func (m *meanSquaredResult) Output() linalg.Vector {
	m.OutputLock.RLock()
	if m.OutputVector != nil {
		m.OutputLock.RUnlock()
		return m.OutputVector
	}
	m.OutputLock.RUnlock()
	m.OutputLock.Lock()
	defer m.OutputLock.Unlock()
	if m.OutputVector != nil {
		return m.OutputVector
	}
	var sum float64
	for i, a := range m.Actual.Output() {
		diff := a - m.Expected[i]
		sum += diff * diff
	}
	m.OutputVector = linalg.Vector{sum}
	return m.OutputVector
}

func (m *meanSquaredResult) Constant(g autofunc.Gradient) bool {
	return m.Actual.Constant(g)
}

func (m *meanSquaredResult) PropagateGradient(upstream linalg.Vector, grad autofunc.Gradient) {
	if !m.Actual.Constant(grad) {
		out := m.Actual.Output()
		upstreamGrad := upstream[0]
		downstream := make(linalg.Vector, len(out))
		for i, a := range out {
			downstream[i] = 2 * upstreamGrad * (a - m.Expected[i])
		}
		m.Actual.PropagateGradient(downstream, grad)
	}
}

// CrossEntropyCost computes the cost using the
// definition of cross entropy.
type CrossEntropyCost struct{}

func (_ CrossEntropyCost) Cost(x linalg.Vector, a autofunc.Result) autofunc.Result {
	return autofunc.Pool(a, func(a autofunc.Result) autofunc.Result {
		xVar := &autofunc.Variable{x}
		logA := autofunc.Log{}.Apply(a)
		oneMinusA := autofunc.AddScaler(autofunc.Scale(a, -1), 1)
		oneMinusX := autofunc.AddScaler(autofunc.Scale(xVar, -1), 1)
		log1A := autofunc.Log{}.Apply(oneMinusA)

		errorVec := autofunc.Add(autofunc.Mul(xVar, logA),
			autofunc.Mul(oneMinusX, log1A))
		return autofunc.Scale(autofunc.SumAll(errorVec), -1)
	})
}

func (_ CrossEntropyCost) CostR(v autofunc.RVector, x linalg.Vector,
	a autofunc.RResult) autofunc.RResult {
	return autofunc.PoolR(a, func(a autofunc.RResult) autofunc.RResult {
		xVar := autofunc.NewRVariable(&autofunc.Variable{x}, autofunc.RVector{})
		logA := autofunc.Log{}.ApplyR(v, a)
		oneMinusA := autofunc.AddScalerR(autofunc.ScaleR(a, -1), 1)
		oneMinusX := autofunc.AddScalerR(autofunc.ScaleR(xVar, -1), 1)
		log1A := autofunc.Log{}.ApplyR(v, oneMinusA)

		errorVec := autofunc.AddR(autofunc.MulR(xVar, logA),
			autofunc.MulR(oneMinusX, log1A))
		return autofunc.ScaleR(autofunc.SumAllR(errorVec), -1)
	})
}

// RegularizingCost adds onto another cost function
// the squared magnitudes of various variables.
type RegularizingCost struct {
	Variables []*autofunc.Variable

	// Penalty is used as a coefficient for the
	// magnitudes of the regularized variables.
	Penalty float64

	CostFunc CostFunc
}

func (r *RegularizingCost) Cost(a linalg.Vector, x autofunc.Result) autofunc.Result {
	regFunc := autofunc.SquaredNorm{}
	cost := r.CostFunc.Cost(a, x)
	for _, variable := range r.Variables {
		norm := regFunc.Apply(variable)
		cost = autofunc.Add(cost, autofunc.Scale(norm, r.Penalty))
	}
	return cost
}

func (r *RegularizingCost) CostR(v autofunc.RVector, a linalg.Vector,
	x autofunc.RResult) autofunc.RResult {
	regFunc := autofunc.SquaredNorm{}
	cost := r.CostFunc.CostR(v, a, x)
	for _, variable := range r.Variables {
		norm := regFunc.ApplyR(v, autofunc.NewRVariable(variable, v))
		cost = autofunc.AddR(cost, autofunc.ScaleR(norm, r.Penalty))
	}
	return cost
}
