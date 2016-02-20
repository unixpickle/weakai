package adaboost

type Sample interface{}

type Classifier interface {
	Classify(s Sample) bool
}
