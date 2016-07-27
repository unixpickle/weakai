// Package boosting implements boosting algorithms such
// as AdaBoost and the more general Gradient Boosting.
package boosting

import "github.com/unixpickle/num-analysis/linalg"

// A SampleList represents an ordered list of training
// samples.
type SampleList interface {
	Len() int
}

// A Classifier classifies samples in a SampleList.
type Classifier interface {
	// Classify returns a vector of classifications,
	// one per sample in the sample list.
	// A classification's sign indicates whether the
	// sample was positive or negative, whereas its
	// magnitude represents the confidence in the
	// classification.
	//
	// The resulting vector belongs to the caller, who
	// may modify the vector.
	Classify(s SampleList) linalg.Vector
}

// SumClassifier classifies samples by adding the
// results of other classifiers.
type SumClassifier struct {
	// Classifiers is a list of classifiers whose
	// outputs are summed in a weighted fashion.
	Classifiers []Classifier

	// Weights contains one weight per classifier, where
	// weights serve a scaling coefficient for a
	// classifier's outputs.
	//
	// This must be the same length as Classifiers.
	Weights []float64
}

func (s *SumClassifier) Classify(list SampleList) linalg.Vector {
	if len(s.Classifiers) == 0 {
		return make(linalg.Vector, list.Len())
	} else if len(s.Classifiers) != len(s.Weights) {
		panic("classifier count must match weight count")
	}
	var res linalg.Vector
	for i, c := range s.Classifiers {
		w := s.Weights[i]
		if res == nil {
			res = c.Classify(list).Scale(w)
		} else {
			res.Add(c.Classify(list).Scale(w))
		}
	}
	return res
}
