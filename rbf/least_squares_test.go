package rbf

import (
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

func TestLeastSquares(t *testing.T) {
	var samples sgd.SliceSampleSet
	for i := 0; i < 100; i++ {
		sample := neuralnet.VectorSample{
			Input: linalg.RandVector(5),
		}
		samples = append(samples, sample)
	}
	net := &Network{
		DistLayer:  NewDistLayerSamples(5, 10, samples),
		ScaleLayer: NewScaleLayerShared(1),
		ExpLayer:   &ExpLayer{Normalize: true},
		OutLayer:   neuralnet.NewDenseLayer(10, 4),
	}
	net.OutLayer.Biases.Var.Vector.Scale(0)
	for i, x := range samples {
		sample := x.(neuralnet.VectorSample)
		sample.Output = net.Apply(&autofunc.Variable{Vector: sample.Input}).Output()
		samples[i] = sample
	}
	expected := net.OutLayer.Weights.Data.Vector
	net.OutLayer = nil
	actual := LeastSquares(net, samples, 3).Weights.Data.Vector
	if actual.Copy().Scale(-1).Add(expected).MaxAbs() > 1e-5 {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}
