package svm

// A Classifier classifies samples as positive or negative using some criterion.
type Classifier interface {
	Classify(sample Sample) bool
}

// A LinearClassifier classifies samples using a hyperplane normal whose pre-image is known.
// This can only be used with solvers that generate a solution which is not inside the transformed
// space represented by the Kernel.
type LinearClassifier struct {
	HyperplaneNormal Sample
	Threshold        float64
	Kernel           Kernel
}

func (c *LinearClassifier) Classify(sample Sample) bool {
	dot := c.Kernel(sample, c.HyperplaneNormal)
	return dot+c.Threshold > 0
}

// A CombinationClassifier classifies novel samples by taking their inner product with a hyperplane
// normal that is a linear combination of support vectors.
// This employs a "kernel trick" to avoid needing to know the actual vector transformation.
type CombinationClassifier struct {
	SupportVectors []Sample
	Coefficients   []float64

	Threshold float64
	Kernel    Kernel
}

func (c CombinationClassifier) Classify(sample Sample) bool {
	var innerProduct float64
	for i, coeff := range c.Coefficients {
		innerProduct += coeff * c.Kernel(c.SupportVectors[i], sample)
	}
	return innerProduct+c.Threshold > 0
}
