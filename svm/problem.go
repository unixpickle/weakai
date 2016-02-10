package svm

// A Sample represents an arbitrary piece of information.
// All samples in a given sample space must have the same number of components.
type Sample []float64

// A Kernel takes two samples from a sample space and computes an inner product between them.
type Kernel func(s1, s2 Sample) float64

// A Problem defines everything needed to build a support vector machine that classifies samples in
// a sample space.
type Problem struct {
	Positives []Sample
	Negatives []Sample
	Kernel    Kernel
}
