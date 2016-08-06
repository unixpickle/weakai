package seqtoseq

import (
	"sort"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

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

// sampleSetSlice converts a sample set into a slice
// of Samples.
func sampleSetSlice(s sgd.SampleSet) []Sample {
	res := make([]Sample, s.Len())
	for i := 0; i < s.Len(); i++ {
		res[i] = s.GetSample(i).(Sample)
	}
	return res
}

// sortSamples sorts the Samples in a SampleSet by size,
// with the longest sequences coming first.
func sortSeqs(s sgd.SampleSet) sgd.SampleSet {
	origSet := sampleSetSlice(s)
	res := make(seqSorter, len(origSet))
	copy(res, origSet)
	sort.Sort(res)

	resSet := make(sgd.SliceSampleSet, len(res))
	for i, x := range res {
		resSet[i] = x
	}
	return resSet
}

type seqSorter []Sample

func (s seqSorter) Len() int {
	return len(s)
}

func (s seqSorter) Less(i, j int) bool {
	return len(s[i].Inputs) > len(s[j].Inputs)
}

func (s seqSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
