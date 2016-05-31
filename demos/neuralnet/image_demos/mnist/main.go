package main

import (
	"log"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	HiddenSize = 300
	LabelCount = 10
	StepSize   = 1e-2

	FilterSize   = 3
	FilterCount  = 5
	FilterStride = 1

	MaxPoolingSpan = 3
)

func main() {
	training := mnist.LoadTrainingDataSet()
	crossValidation := mnist.LoadTestingDataSet()

	log.Println("Printing initial scores...")

	net, trainer := createNetAndTrainer(training)
	printScore("Initial training", net, training)
	printScore("Initial cross", net, crossValidation)
	for {
		trainer.Train(net)
		printScore("Training", net, training)
		printScore("Cross", net, crossValidation)
	}
}

func createNetAndTrainer(d mnist.DataSet) (*neuralnet.Network, *neuralnet.SGD) {
	convOutWidth := (d.Width-FilterSize)/FilterStride + 1
	convOutHeight := (d.Height-FilterSize)/FilterStride + 1

	poolOutWidth := convOutWidth / MaxPoolingSpan
	if convOutWidth%MaxPoolingSpan != 0 {
		poolOutWidth++
	}
	poolOutHeight := convOutWidth / MaxPoolingSpan
	if convOutHeight%MaxPoolingSpan != 0 {
		poolOutHeight++
	}

	net, _ := neuralnet.NewNetwork([]neuralnet.LayerPrototype{
		&neuralnet.ConvParams{
			Activation:   neuralnet.Sigmoid{},
			FilterCount:  FilterCount,
			FilterWidth:  FilterSize,
			FilterHeight: FilterSize,
			Stride:       FilterStride,
			InputWidth:   d.Width,
			InputHeight:  d.Height,
			InputDepth:   1,
		},
		&neuralnet.MaxPoolingParams{
			XSpan:       MaxPoolingSpan,
			YSpan:       MaxPoolingSpan,
			InputWidth:  convOutWidth,
			InputHeight: convOutHeight,
			InputDepth:  FilterCount,
		},
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  poolOutWidth * poolOutHeight * FilterCount,
			OutputCount: HiddenSize,
		},
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  HiddenSize,
			OutputCount: LabelCount,
		},
	})
	net.Randomize()

	trainer := &neuralnet.SGD{
		CostFunc: neuralnet.MeanSquaredCost{},
		Inputs:   d.IntensityVectors(),
		Outputs:  d.LabelVectors(),
		StepSize: StepSize,
		Epochs:   1,
	}

	return net, trainer
}

func printScore(prefix string, n *neuralnet.Network, d mnist.DataSet) {
	classifier := func(v []float64) int {
		n.SetInput(v)
		n.PropagateForward()
		return networkOutput(n)
	}
	correctCount := d.NumCorrect(classifier)
	histogram := d.CorrectnessHistogram(classifier)
	log.Printf("%s: %d/%d - %s", prefix, correctCount, len(d.Samples), histogram)
}

func networkOutput(n *neuralnet.Network) int {
	out := n.Output()
	var maxIdx int
	var max float64
	for i, x := range out {
		if i == 0 || x > max {
			max = x
			maxIdx = i
		}
	}
	return maxIdx
}
