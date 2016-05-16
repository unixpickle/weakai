package main

import (
	"fmt"
	"log"
	"os"

	"github.com/unixpickle/weakai/neuralnet"
)

const (
	HiddenSize = 300
	LabelCount = 10
	StepSize   = 1e-2

	SampleCount = 6000
	TrainCount  = 3500

	FilterSize   = 3
	FilterCount  = 5
	FilterStride = 1

	MaxPoolingSpan = 3
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage: mnist <labels> <samples>")
		os.Exit(1)
	}

	labels, err := ReadLabels(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read labels:", err)
		os.Exit(1)
	}

	samples, err := ReadSamples(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read samples:", err)
		os.Exit(1)
	}

	log.Printf("Read %d samples", len(samples))

	trainNetwork(labels[:SampleCount], samples[:SampleCount])
}

func trainNetwork(labels []int, samples []*neuralnet.Tensor3) {
	convOutWidth := (samples[0].Width-FilterSize)/FilterStride + 1
	convOutHeight := (samples[0].Height-FilterSize)/FilterStride + 1

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
			InputWidth:   samples[0].Width,
			InputHeight:  samples[0].Height,
			InputDepth:   samples[0].Depth,
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

	inputList := make([][]float64, len(samples))
	outputList := make([][]float64, len(samples))
	for i, x := range samples {
		inputList[i] = x.Data
		out := make([]float64, LabelCount)
		out[labels[i]] = 1
		outputList[i] = out
	}

	trainer := &neuralnet.SGD{
		CostFunc: neuralnet.MeanSquaredCost{},
		Inputs:   inputList[:TrainCount],
		Outputs:  outputList[:TrainCount],
		StepSize: StepSize,
		Epochs:   1,
	}

	// Used only for input/output fields.
	crossTrainer := &neuralnet.SGD{
		Inputs:  inputList[TrainCount:],
		Outputs: outputList[TrainCount:],
	}

	log.Printf("Initial score: %d/%d (%d/%d cross)",
		networkCorrectCount(trainer, net), TrainCount,
		networkCorrectCount(crossTrainer, net), SampleCount-TrainCount)

	for {
		trainer.Train(net)
		log.Printf("New score: %d/%d (%d/%d cross)",
			networkCorrectCount(trainer, net), TrainCount,
			networkCorrectCount(crossTrainer, net), SampleCount-TrainCount)
	}
}

func networkCorrectCount(t *neuralnet.SGD, n *neuralnet.Network) int {
	var count int
	for i, sample := range t.Inputs {
		n.SetInput(sample)
		n.PropagateForward()
		out := networkOutput(n)
		if t.Outputs[i][out] == 1 {
			count++
		}
	}
	return count
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
