package rbm

import (
	"math/rand"
	"runtime"
)

// A Trainer stores parameters for training an RBM.
type Trainer struct {
	GibbsSteps int
	StepSize   float64
	Epochs     int
	BatchSize  int
}

// Train trains the RBM for the supplied inputs.
func (t *Trainer) Train(r *RBM, inputs [][]bool) {
	procCount := runtime.GOMAXPROCS(0)
	inputChan := make(chan []bool, t.BatchSize)
	outputChan := make(chan *RBMGradient, t.BatchSize)

	defer close(inputChan)

	for i := 0; i < procCount; i++ {
		go func() {
			for input := range inputChan {
				grad := r.LogLikelihoodGradient([][]bool{input}, t.GibbsSteps)
				outputChan <- grad
			}
		}()
	}

	for i := 0; i < t.Epochs; i++ {
		perm := rand.Perm(len(inputs))
		for j := 0; j < len(inputs); j += t.BatchSize {
			batchCount := t.BatchSize
			if j+batchCount > len(inputs) {
				j = len(inputs) - batchCount
			}
			for k := j; k < j+batchCount; k++ {
				inputChan <- inputs[perm[j]]
			}
			var batch *RBMGradient
			for k := 0; k < batchCount; k++ {
				grad := <-outputChan
				if batch == nil {
					batch = grad
				} else {
					batch.HiddenBiases.Add(grad.HiddenBiases)
					batch.VisibleBiases.Add(grad.VisibleBiases)
					batch.Weights.Add(grad.Weights)
				}
			}
			r.HiddenBiases.Add(batch.HiddenBiases.Scale(t.StepSize))
			r.VisibleBiases.Add(batch.VisibleBiases.Scale(t.StepSize))
			r.Weights.Add(batch.Weights.Scale(t.StepSize))
		}
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
