package svm

import (
	"math"
	"sort"
)

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
	sampleSum := make([]float64, len(c.SupportVectors[0].V))
	for i, vec := range c.SupportVectors {
		coeff := c.Coefficients[i]
		for j := range sampleSum {
			sampleSum[j] += coeff * vec.V[j]
		}
	}
	return &LinearClassifier{
		Kernel:           c.Kernel,
		HyperplaneNormal: Sample{V: sampleSum},
		Threshold:        c.Threshold,
	}
}

func (c *CombinationClassifier) computeThreshold(p *Problem) {
	sampleProducts := make([]float64, 0, len(p.Positives)+len(p.Negatives))
	for _, pos := range p.Positives {
		product := c.sampleProduct(pos)
		sampleProducts = append(sampleProducts, product)
	}
	for _, neg := range p.Negatives {
		product := c.sampleProduct(neg)
		sampleProducts = append(sampleProducts, product)
	}

	sortedProducts := make([]float64, len(sampleProducts))
	copy(sortedProducts, sampleProducts)
	sort.Float64s(sortedProducts)

	var minError float64
	var minErrorThreshold float64

	for i := 0; i < len(sortedProducts)-1; i++ {
		thresh := (sortedProducts[i] + sortedProducts[i+1]) / 2
		var err float64
		for j := range p.Positives {
			err += math.Max(0, 1-(sampleProducts[j]-thresh))
		}
		for j := range p.Negatives {
			err += math.Max(0, 1+(sampleProducts[j+len(p.Positives)]-thresh))
		}
		if err < minError || i == 0 {
			minError = err
			minErrorThreshold = thresh
		}
	}

	c.Threshold = -minErrorThreshold
}

func (c *CombinationClassifier) sampleProduct(sample Sample) float64 {
	var innerProduct float64
	for i, coeff := range c.Coefficients {
		innerProduct += coeff * c.Kernel(c.SupportVectors[i], sample)
	}
	return innerProduct
}
