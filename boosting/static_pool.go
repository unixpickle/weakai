package boosting

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/unixpickle/num-analysis/linalg"
)

// A StaticPool stores an unchanging list of classifiers
// and caches their outputs on an unchanging sample set.
type StaticPool struct {
	classifiers  []Classifier
	outputMatrix blas64.General
}

// NewStaticPool creates a static pool for the given
// classifiers and samples.
// In the process of creating the pool, every classifier
// is run on every sample.
// As a result, NewStaticPool may take some time to run.
func NewStaticPool(c []Classifier, s SampleList) *StaticPool {
	res := &StaticPool{
		classifiers: make([]Classifier, len(c)),
		outputMatrix: blas64.General{
			Rows:   len(c),
			Cols:   s.Len(),
			Stride: s.Len(),
			Data:   make([]float64, len(c)*s.Len()),
		},
	}
	copy(res.classifiers, c)

	var rowIdx int
	for _, classifier := range c {
		output := classifier.Classify(s)
		copy(res.outputMatrix.Data[rowIdx:], output)
		rowIdx += s.Len()
	}

	return res
}

// BestClassifier returns the classifier in the static
// pool whose output correlates the most highly with
// the given weight vector, as measured by absolute
// cosine distance.
//
// The list argument is ignored, since a StaticPool
// always uses the set of samples it was given when
// it was initialized.
func (s *StaticPool) BestClassifier(list SampleList, weights linalg.Vector) Classifier {
	vec := blas64.Vector{
		Inc:  1,
		Data: weights,
	}
	output := blas64.Vector{
		Inc:  1,
		Data: make([]float64, len(s.classifiers)),
	}
	blas64.Gemv(blas.NoTrans, 1, s.outputMatrix, vec, 0, output)
	largest := blas64.Iamax(len(s.classifiers), output)
	return s.classifiers[largest]
}
