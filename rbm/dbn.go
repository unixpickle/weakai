package rbm

import "math/rand"

// A DBN is a Deep Belief Network, which is
// nothing more than stacked RBMs.
type DBN []*RBM

// Sample samples an output vector given an
// input vector.
// If r is nil, this uses the rand package's
// default generator.
func (d DBN) Sample(r *rand.Rand, input []bool) []bool {
	currentInput := input
	for _, layer := range d {
		output := make([]bool, len(layer.HiddenBiases))
		layer.SampleHidden(r, output, currentInput)
		currentInput = output
	}
	return currentInput
}

// SampleInput samples an input vector given
// an output vector.
// If r is nil, this uses the rand package's
// default generator.
func (d DBN) SampleInput(r *rand.Rand, output []bool) []bool {
	currentOutput := output
	for i := len(d) - 1; i >= 0; i-- {
		layer := d[i]
		input := make([]bool, len(layer.VisibleBiases))
		layer.SampleVisible(r, input, currentOutput)
		currentOutput = input
	}
	return currentOutput
}
