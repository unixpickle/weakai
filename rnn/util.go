package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

func joinVectors(v []linalg.Vector) linalg.Vector {
	var totalSize int
	for _, x := range v {
		totalSize += len(x)
	}
	res := make(linalg.Vector, totalSize)
	return joinVectorsInPlace(res, v)
}

func joinVectorsInPlace(dest linalg.Vector, v []linalg.Vector) linalg.Vector {
	var idx int
	for _, x := range v {
		copy(dest[idx:], x)
		idx += len(x)
	}
	return dest
}

func joinBlockInput(in *BlockInput) autofunc.Result {
	results := make([]autofunc.Result, 0, len(in.States)*2)
	for i := range in.States {
		results = append(results, in.Inputs[i], in.States[i])
	}
	return autofunc.Concat(results...)
}

func joinBlockRInput(in *BlockRInput) autofunc.RResult {
	results := make([]autofunc.RResult, 0, len(in.States)*2)
	for i := range in.States {
		results = append(results, in.Inputs[i], in.States[i])
	}
	return autofunc.ConcatR(results...)
}

func joinVariables(vars []*autofunc.Variable) autofunc.Result {
	results := make([]autofunc.Result, len(vars))
	for i, v := range vars {
		results[i] = v
	}
	return autofunc.Concat(results...)
}

func joinRVariables(vars []*autofunc.RVariable) autofunc.RResult {
	results := make([]autofunc.RResult, len(vars))
	for i, v := range vars {
		results[i] = v
	}
	return autofunc.ConcatR(results...)
}

func splitVectors(v linalg.Vector, n int) []linalg.Vector {
	if len(v)%n != 0 {
		panic("length is not properly divisible")
	}
	partLen := len(v) / n
	idx := 0
	res := make([]linalg.Vector, n)
	for i := range res {
		res[i] = v[idx : idx+partLen]
		idx += partLen
	}
	return res
}

func evalCostFuncDeriv(c neuralnet.CostFunc, expected, actual linalg.Vector) linalg.Vector {
	variable := &autofunc.Variable{Vector: actual}
	result := make(linalg.Vector, len(actual))
	res := c.Cost(expected, variable)
	res.PropagateGradient([]float64{1}, autofunc.Gradient{variable: result})
	return result
}
