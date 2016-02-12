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

func (c *CombinationClassifier) Classify(sample Sample) bool {
	return c.sampleProduct(sample)+c.Threshold > 0
}

// Linearize converts a CombinationClassifier into a LinearClassifier, assuming that the underlying
// kernel is LinearKernel.
// This will not work for non-linear kernels.
func (c *CombinationClassifier) Linearize() *LinearClassifier {
	sampleSum := make(Sample, len(c.SupportVectors[0]))
	for i, vec := range c.SupportVectors {
		coeff := c.Coefficients[i]
		for j := range sampleSum {
			sampleSum[j] += coeff * vec[j]
		}
	}
	return &LinearClassifier{
		Kernel:           c.Kernel,
		HyperplaneNormal: sampleSum,
		Threshold:        c.Threshold,
	}
}

func (c *CombinationClassifier) computeThreshold(p *Problem) {
	var lowestPositive float64
	var highestNegative float64
	for i, pos := range p.Positives {
		product := c.sampleProduct(pos)
		if product < lowestPositive || i == 0 {
			lowestPositive = product
		}
	}
	for i, neg := range p.Negatives {
		product := c.sampleProduct(neg)
		if product > highestNegative || i == 0 {
			highestNegative = product
		}
	}
	c.Threshold = -(lowestPositive + highestNegative) / 2
}

func (c *CombinationClassifier) sampleProduct(sample Sample) float64 {
	var innerProduct float64
	for i, coeff := range c.Coefficients {
		innerProduct += coeff * c.Kernel(c.SupportVectors[i], sample)
	}
	return innerProduct
}
