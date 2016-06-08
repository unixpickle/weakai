package neuralnet

import (
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

// MeanSquaredCost computes the cost as 1/2*||a-x||^2
// where a is the actual output and x is the desired
// output.
type MeanSquaredCost struct{}

func (_ MeanSquaredCost) Cost(a linalg.Vector, x autofunc.Result) autofunc.Result {
	aVar := &autofunc.Variable{a.Copy().Scale(-1)}
	return autofunc.SquaredNorm{}.Apply(autofunc.Add(aVar, x))
}

func (_ MeanSquaredCost) CostR(v autofunc.RVector, a linalg.Vector,
	x autofunc.RResult) autofunc.RResult {
	aVar := &autofunc.Variable{a.Copy().Scale(-1)}
	aVarR := autofunc.NewRVariable(aVar, v)
	return autofunc.SquaredNorm{}.ApplyR(v, autofunc.AddR(aVarR, x))
}

// CrossEntropyCost computes the cost using the
// definition of cross entropy.
type CrossEntropyCost struct{}

func (_ CrossEntropyCost) Cost(a linalg.Vector, inX autofunc.Result) autofunc.Result {
	return autofunc.Pool(inX, func(x autofunc.Result) autofunc.Result {
		aVar := &autofunc.Variable{a}
		logA := autofunc.Log{}.Apply(aVar)
		oneMinusA := autofunc.AddScaler(autofunc.Scale(aVar, -1), 1)
		oneMinusX := autofunc.AddScaler(autofunc.Scale(x, -1), 1)
		log1A := autofunc.Log{}.Apply(oneMinusA)

		errorVec := autofunc.Add(autofunc.Mul(aVar, logA),
			autofunc.Mul(oneMinusX, log1A))
		return autofunc.Scale(autofunc.SumAll(errorVec), -1)
	})
}

func (_ CrossEntropyCost) CostR(v autofunc.RVector, a linalg.Vector,
	inX autofunc.RResult) autofunc.RResult {
	return autofunc.PoolR(inX, func(x autofunc.RResult) autofunc.RResult {
		aVar := autofunc.NewRVariable(&autofunc.Variable{a}, autofunc.RVector{})
		logA := autofunc.Log{}.ApplyR(v, aVar)
		oneMinusA := autofunc.AddScalerR(autofunc.ScaleR(aVar, -1), 1)
		oneMinusX := autofunc.AddScalerR(autofunc.ScaleR(x, -1), 1)
		log1A := autofunc.Log{}.ApplyR(v, oneMinusA)

		errorVec := autofunc.AddR(autofunc.MulR(aVar, logA),
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
