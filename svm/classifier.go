package svm

// A Classifier uses inner products to determine if novel samples are likely to be positive or not.
type Classifier struct {
	HyperplaneNormal Sample
	Threshold        float64
	Kernel           Kernel
}

// Classify returns true if the given sample is classified as positive.
func (c *Classifier) Classify(sample Sample) bool {
	dot := c.Kernel(sample, c.HyperplaneNormal)
	return dot+c.Threshold > 0
}
