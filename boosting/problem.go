package boosting

type Problem struct {
	Classifiers     []Classifier
	Samples         []Sample
	Classifications []bool
}

func (p *Problem) numCorrect(s *Solution) int {
	correct := 0
	for i, sample := range p.Samples {
		if s.Classify(sample) == p.Classifications[i] {
			correct++
		}
	}
	return correct
}
