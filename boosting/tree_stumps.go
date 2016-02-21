package boosting

import "sort"

// A TreeStump is a binary classifier for Samples which are arrays of float64s.
// The TreeStump accepts all samples whose i-th value is greater than or equal to a threshold.
type TreeStump struct {
	FieldIndex int
	Threshold  float64
}

// AllTreeStumps generates every tree stump that separates data in the given sample space.
func AllTreeStumps(samples []Sample, dimensions int) []*TreeStump {
	stumps := make([]*TreeStump, 0, len(samples)*dimensions)
	for d := 0; d < dimensions; d++ {
		values := make([]float64, 0, len(samples))
		seenValues := map[float64]bool{}
		for _, s := range samples {
			val := s.([]float64)[d]
			if !seenValues[val] {
				seenValues[val] = true
				values = append(values, val)
			}
		}
		sort.Float64s(values)
		for i, val := range values {
			if i == 0 {
				stumps = append(stumps, &TreeStump{FieldIndex: d, Threshold: val - 1})
			} else {
				lastVal := values[i-1]
				average := (lastVal + val) / 2
				stumps = append(stumps, &TreeStump{FieldIndex: d, Threshold: average})
			}
		}
	}
	return stumps
}

// Classify treats the sample as a []float64 and checks that a specific entry in it exceeds a
// threshold.
func (t *TreeStump) Classify(s Sample) bool {
	vec := s.([]float64)
	return vec[t.FieldIndex] >= t.Threshold
}
