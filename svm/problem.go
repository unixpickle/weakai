package svm

// A Sample represents an arbitrary piece of information.
// All samples in a given sample space must have the same number of components.
type Sample struct {
	V []float64

	// UserInfo can be used by a Kernel to uniquely identify a given Sample.
	// If a solver generates its own Samples, said Samples will have UserInfo set to 0.
	UserInfo int
}

// A Kernel takes two samples from a sample space and computes an inner product between them.
type Kernel func(s1, s2 Sample) float64

// A Problem defines everything needed to build a support vector machine that classifies samples in
// a sample space.
type Problem struct {
	Positives []Sample
	Negatives []Sample
	Kernel    Kernel
}
