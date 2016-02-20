package adaboost

type Solution struct {
	Classifiers []Classifier
	Weights     []float64
}

// PartialSolution returns a Solution that uses the first n classifiers of this solution.
func (s *Solution) PartialSolution(n int) *Solution {
	return &Solution{
		Classifiers: s.Classifiers[:n],
		Weights:     s.Weights[:n],
	}
}

func (s *Solution) Classify(sam Sample) bool {
	return s.Evaluate(sam) >= 0
}

func (s *Solution) Evaluate(sam Sample) float64 {
	var sum float64
	for i, classifier := range s.Classifiers {
		res := classifier.Classify(sam)
		if res {
			sum += s.Weights[i]
		} else {
			sum -= s.Weights[i]
		}
	}
	return sum
}
