package rnn

import (
	"sort"

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

func costFuncDeriv(c neuralnet.CostFunc, expected, actual linalg.Vector) linalg.Vector {
	variable := &autofunc.Variable{Vector: actual}
	result := make(linalg.Vector, len(actual))
	res := c.Cost(expected, variable)
	res.PropagateGradient([]float64{1}, autofunc.Gradient{variable: result})
	return result
}

func costFuncRDeriv(c neuralnet.CostFunc, expected, actual,
	actualR linalg.Vector) (deriv, rDeriv linalg.Vector) {
	variable := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: actual},
		ROutputVec: actualR,
	}
	deriv = make(linalg.Vector, len(actual))
	rDeriv = make(linalg.Vector, len(actual))
	res := c.CostR(autofunc.RVector{}, expected, variable)
	res.PropagateRGradient([]float64{1}, []float64{0},
		autofunc.RGradient{variable.Variable: rDeriv},
		autofunc.Gradient{variable.Variable: deriv})
	return
}

// sampleSetSequences converts a sample set into a
// list of Sequences.
func sampleSetSequences(s neuralnet.SampleSet) []Sequence {
	res := make([]Sequence, s.Len())
	for i := 0; i < s.Len(); i++ {
		res[i] = s.GetSample(i).(Sequence)
	}
	return res
}

// sortSeqs sorts the Sequences in a SampleSet by size,
// with the longest sequences coming first.
func sortSeqs(s neuralnet.SampleSet) neuralnet.SampleSet {
	origSet := sampleSetSequences(s)
	res := make(seqSorter, len(origSet))
	copy(res, origSet)
	sort.Sort(res)

	resSet := make(neuralnet.SliceSampleSet, len(res))
	for i, x := range res {
		resSet[i] = x
	}
	return resSet
}

type seqSorter []Sequence

func (s seqSorter) Len() int {
	return len(s)
}

func (s seqSorter) Less(i, j int) bool {
	return len(s[i].Inputs) > len(s[j].Inputs)
}

func (s seqSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
