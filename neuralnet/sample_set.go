package neuralnet

import (
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
)

// SampleSet is a set of input samples and their
// corresponding expected outputs.
type SampleSet struct {
	Inputs  []linalg.Vector
	Outputs []linalg.Vector
}

// Copy creates a shallow copy of this set.
// The copy will have new input and output
// slices, but the vectors within the slices
// will be the same.
func (s *SampleSet) Copy() *SampleSet {
	res := &SampleSet{
		Inputs:  make([]linalg.Vector, len(s.Inputs)),
		Outputs: make([]linalg.Vector, len(s.Outputs)),
	}
	copy(res.Inputs, s.Inputs)
	copy(res.Outputs, s.Outputs)
	return res
}

// Shuffle randomly re-orders the set of samples.
func (s *SampleSet) Shuffle() {
	oldInputs := make([]linalg.Vector, len(s.Inputs))
	oldOutputs := make([]linalg.Vector, len(s.Outputs))
	copy(oldInputs, s.Inputs)
	copy(oldOutputs, s.Outputs)

	perm := rand.Perm(len(oldInputs))
	for i, x := range perm {
		s.Inputs[i] = oldInputs[x]
		s.Outputs[i] = oldOutputs[x]
	}
}

// Split splits the SampleSet into two sample sets.
// The first sample set will contain the first l
// samples in the original sample set.
// The l argument must not be greater than the total
// number of samples.
func (s *SampleSet) Split(l int) (*SampleSet, *SampleSet) {
	return &SampleSet{s.Inputs[:l], s.Outputs[:l]},
		&SampleSet{s.Inputs[l:], s.Outputs[l:]}
}

// Subset returns the set of samples in the range
// between index1 (inclusive) and index2 (exclusive).
func (s *SampleSet) Subset(index1, index2 int) *SampleSet {
	return &SampleSet{
		Inputs:  s.Inputs[index1:index2],
		Outputs: s.Outputs[index1:index2],
	}
}
