package boosting

type Sample interface{}

type Classifier interface {
	Classify(s Sample) bool
}
