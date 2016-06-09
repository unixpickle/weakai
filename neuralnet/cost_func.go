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
// set of samples.
func TotalCost(c CostFunc, layer autofunc.Func, s *SampleSet) float64 {
	var totalCost float64
	for i, input := range s.Inputs {
		inVar := &autofunc.Variable{input}
		result := layer.Apply(inVar)
		costOut := c.Cost(s.Outputs[i], result)
		totalCost += costOut.Output()[0]
	}
	return totalCost
}

// MeanSquaredCost computes the cost as ||a-x||^2
// where a is the actual output and x is the desired
// output.
type MeanSquaredCost struct {
	Cache *autofunc.VectorCache
}

func (m MeanSquaredCost) Cost(x linalg.Vector, a autofunc.Result) autofunc.Result {
	return &meanSquaredResult{
		Cache:    m.Cache,
		Actual:   a,
		Expected: x,
	}
}

func (m MeanSquaredCost) CostR(v autofunc.RVector, a linalg.Vector,
	x autofunc.RResult) autofunc.RResult {
	arith := autofunc.Arithmetic{m.Cache}

	// TODO: avoid the .Copy() call.
	aVar := &autofunc.Variable{a.Copy().Scale(-1)}

	aVarR := autofunc.NewRVariableCache(aVar, v, m.Cache)
	return autofunc.SquaredNorm{m.Cache}.ApplyR(v, arith.AddR(aVarR, x))
}

type meanSquaredResult struct {
	Cache *autofunc.VectorCache

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
		downstream := m.Cache.Alloc(len(out))
		for i, a := range out {
			downstream[i] = 2 * upstreamGrad * (a - m.Expected[i])
		}
		m.Actual.PropagateGradient(downstream, grad)
		m.Cache.Free(downstream)
	}
}

func (m *meanSquaredResult) Release() {
	m.Actual.Release()
}

// CrossEntropyCost computes the cost using the
// definition of cross entropy.
type CrossEntropyCost struct {
	Cache *autofunc.VectorCache
}

func (c CrossEntropyCost) Cost(a linalg.Vector, inX autofunc.Result) autofunc.Result {
	return autofunc.Pool(inX, func(x autofunc.Result) autofunc.Result {
		arith := autofunc.Arithmetic{c.Cache}

		aVar := &autofunc.Variable{a}
		logA := autofunc.Log{c.Cache}.Apply(aVar)
		oneMinusA := arith.AddScaler(arith.Scale(aVar, -1), 1)
		oneMinusX := arith.AddScaler(arith.Scale(x, -1), 1)
		log1A := autofunc.Log{c.Cache}.Apply(oneMinusA)

		errorVec := arith.Add(arith.Mul(aVar, logA),
			arith.Mul(oneMinusX, log1A))
		return arith.Scale(arith.SumAll(errorVec), -1)
	})
}

func (c CrossEntropyCost) CostR(v autofunc.RVector, a linalg.Vector,
	inX autofunc.RResult) autofunc.RResult {
	return autofunc.PoolR(inX, func(x autofunc.RResult) autofunc.RResult {
		arith := autofunc.Arithmetic{c.Cache}

		aVar := autofunc.NewRVariableCache(&autofunc.Variable{a}, autofunc.RVector{}, c.Cache)
		logA := autofunc.Log{c.Cache}.ApplyR(v, aVar)
		oneMinusA := arith.AddScalerR(arith.ScaleR(aVar, -1), 1)
		oneMinusX := arith.AddScalerR(arith.ScaleR(x, -1), 1)
		log1A := autofunc.Log{c.Cache}.ApplyR(v, oneMinusA)

		errorVec := arith.AddR(arith.MulR(aVar, logA),
			arith.MulR(oneMinusX, log1A))
		return arith.ScaleR(arith.SumAllR(errorVec), -1)
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

	Cache *autofunc.VectorCache
}

func (r *RegularizingCost) Cost(a linalg.Vector, x autofunc.Result) autofunc.Result {
	arith := autofunc.Arithmetic{r.Cache}
	regFunc := autofunc.SquaredNorm{r.Cache}
	cost := r.CostFunc.Cost(a, x)
	for _, variable := range r.Variables {
		norm := regFunc.Apply(variable)
		cost = arith.Add(cost, arith.Scale(norm, r.Penalty))
	}
	return cost
}

func (r *RegularizingCost) CostR(v autofunc.RVector, a linalg.Vector,
	x autofunc.RResult) autofunc.RResult {
	arith := autofunc.Arithmetic{r.Cache}
	regFunc := autofunc.SquaredNorm{r.Cache}
	cost := r.CostFunc.CostR(v, a, x)
	for _, variable := range r.Variables {
		rVar := autofunc.NewRVariableCache(variable, v, r.Cache)
		norm := regFunc.ApplyR(v, rVar)
		cost = arith.AddR(cost, arith.ScaleR(norm, r.Penalty))
	}
	return cost
}
