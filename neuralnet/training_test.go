package neuralnet

import (
	"math"
	"math/rand"
	"testing"
)

func TestTrainingXOR(t *testing.T) {
	net, err := NewNetwork([]LayerPrototype{
		&DenseParams{
			Activation:  Sigmoid{},
			InputCount:  2,
			OutputCount: 4,
		},
		&DenseParams{
			Activation:  Sigmoid{},
			InputCount:  4,
			OutputCount: 1,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	trainer := &SGD{
		CostFunc: MeanSquaredCost{},
		Inputs: [][]float64{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1},
		},
		Outputs:  [][]float64{{0}, {1}, {1}, {0}},
		StepSize: 0.9,
		Epochs:   100000,

		// Adversarial batch size ensures that the batch
		// needn't be 1 and needn't divide the len(Inputs).
		BatchSize: 3,
	}

	rand.Seed(123123)
	net.Randomize()
	trainer.Train(net)

	for i, sample := range trainer.Inputs {
		net.SetInput(sample)
		net.PropagateForward()
		expected := trainer.Outputs[i][0]
		actual := net.Output()[0]
		if math.Abs(expected-actual) > 2e-2 {
			t.Errorf("expected %f for input %v but got %f", expected, sample, actual)
		}
	}
}
