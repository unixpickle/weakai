package svm

// LinearKernel is a Kernel that returns the straight dot product of the two input samples.
func LinearKernel(s1, s2 Sample) float64 {
	if len(s1) != len(s2) {
		panic("samples must be of the sample dimension")
	}
	var sum float64
	for i, x := range s1 {
		sum += x * s2[i]
	}
	return sum
}
