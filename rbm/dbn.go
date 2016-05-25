package rbm

// A DBN is a Deep Belief Network, which is
// nothing more than stacked RBMs.
type DBN []*RBM

// Sample samples an output vector given an
// input vector.
func (d DBN) Sample(input []bool) []bool {
	currentInput := input
	for _, layer := range d {
		output := make([]bool, len(layer.HiddenBiases))
		layer.SampleHidden(output, currentInput)
		currentInput = output
	}
	return currentInput
}

// SampleInput samples an input vector given
// an output vector.
func (d DBN) SampleInput(output []bool) []bool {
	currentOutput := output
	for i := len(d) - 1; i >= 0; i-- {
		layer := d[i]
		input := make([]bool, len(layer.VisibleBiases))
		layer.SampleVisible(input, currentOutput)
		currentOutput = input
	}
	return currentOutput
}
