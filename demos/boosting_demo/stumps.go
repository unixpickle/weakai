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
	Sign       float64
}

func (t TreeStump) Classify(s boosting.SampleList) linalg.Vector {
	l := s.(SampleList)
	res := make(linalg.Vector, s.Len())
	for i, sample := range l {
		if sample[t.FieldIndex] >= t.Threshold {
			res[i] = t.Sign
		} else {
			res[i] = -t.Sign
		}
	}
	return res
}

type StumpPool []TreeStump

func NewStumpPool(samples SampleList) StumpPool {
	dims := len(samples[0])
	res := make(StumpPool, 0, len(samples)*dims*2)
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
				t = TreeStump{FieldIndex: d, Threshold: val - 1, Sign: 1}
			} else {
				lastVal := values[i-1]
				average := (lastVal + val) / 2
				t = TreeStump{FieldIndex: d, Threshold: average, Sign: 1}
			}
			res = append(res, t)
			t.Sign *= -1
			res = append(res, t)
		}
	}
	return res
}

func (s StumpPool) BestClassifier(l boosting.SampleList, w linalg.Vector) boosting.Classifier {
	var bestDot float64
	var bestC boosting.Classifier
	for i, c := range s {
		dot := c.Classify(l).Dot(w)
		if i == 0 || dot > bestDot {
			bestDot = dot
			bestC = c
		}
	}
	return bestC
}
