package rbm

// A Trainer stores parameters for training an RBM.
type Trainer struct {
	GibbsSteps int
	StepSize   float64
	Epochs     int
}

// Train trains the RBM for the supplied inputs.
func (t *Trainer) Train(r *RBM, inputs [][]bool) {
	for i := 0; i < t.Epochs; i++ {
		grad := r.LogLikelihoodGradient(inputs, t.GibbsSteps)
		r.HiddenBiases.Add(grad.HiddenBiases.Scale(t.StepSize))
		r.VisibleBiases.Add(grad.VisibleBiases.Scale(t.StepSize))
		r.Weights.Add(grad.Weights.Scale(t.StepSize))
	}
}

// TrainDeep performs pre-training on a DBN (i.e.
// a bunch of stacked RBMs).
// The layers are ordered from the input layer to
// the output layer.
func (t *Trainer) TrainDeep(layers []*RBM, inputs [][]bool) {
	layerInputs := inputs
	for _, layer := range layers {
		t.Train(layer, layerInputs)
		newInputs := make([][]bool, len(layerInputs))
		for i, input := range layerInputs {
			newInputs[i] = make([]bool, len(layer.HiddenBiases))
			layer.SampleHidden(newInputs[i], input)
		}
		layerInputs = newInputs
	}
}
