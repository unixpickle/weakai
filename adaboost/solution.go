package adaboost

type Solution struct {
	Classifiers []Classifier
	Weights     []float64
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
