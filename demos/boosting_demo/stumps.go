package main

import (
	"sort"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/boosting"
)

// A TreeStump makes classifications based on a vector
// component being greater than a certain threshold.
type TreeStump struct {
	FieldIndex int
	Threshold  float64
}

func (t TreeStump) Classify(s boosting.SampleList) linalg.Vector {
	l := s.(SampleList)
	res := make(linalg.Vector, s.Len())
	for i, sample := range l {
		if sample[t.FieldIndex] >= t.Threshold {
			res[i] = 1
		} else {
			res[i] = -1
		}
	}
	return res
}

func StumpPool(samples SampleList) boosting.Pool {
	dims := len(samples[0])
	res := make([]boosting.Classifier, 0, len(samples)*dims)
	for d := 0; d < dims; d++ {
		values := make([]float64, 0, len(samples))
		seenValues := map[float64]bool{}
		for _, s := range samples {
			val := s[d]
			if !seenValues[val] {
				seenValues[val] = true
				values = append(values, val)
			}
		}
		sort.Float64s(values)
		for i, val := range values {
			var t TreeStump
			if i == 0 {
				t = TreeStump{FieldIndex: d, Threshold: val - 1}
			} else {
				lastVal := values[i-1]
				average := (lastVal + val) / 2
				t = TreeStump{FieldIndex: d, Threshold: average}
			}
			res = append(res, t)
		}
	}
	return boosting.NewStaticPool(res, samples)
}
