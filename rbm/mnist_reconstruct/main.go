package main

import (
	"log"
	"runtime"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/weakai/rbm"
)

const (
	ImageSize     = 28 * 28
	GibbsSteps    = 10
	BigStepSize   = 1e-2
	BigEpochs     = 100
	SmallStepSize = 1e-3
	SmallEpochs   = 100

	ReconstructionGridSize = 5
)

var HiddenSizes = []int{300, 100}

func main() {
	training := mnist.LoadTrainingDataSet()
	samples := make([][]bool, len(training.Samples))
	for i, sample := range training.Samples {
		samples[i] = make([]bool, len(sample.Intensities))
		for j, x := range sample.Intensities {
			if x > 0.5 {
				samples[i][j] = true
			}
		}
	}

	layers := buildLayers()
	trainer := rbm.Trainer{
		GibbsSteps: GibbsSteps,
		StepSize:   BigStepSize,
		Epochs:     BigEpochs,
		BatchSize:  runtime.GOMAXPROCS(0),
	}
	log.Println("Training...")
	trainer.TrainDeep(layers, samples[:1000])
	trainer.StepSize = SmallStepSize
	trainer.Epochs = SmallEpochs
	trainer.TrainDeep(layers, samples[:1000])
	log.Println("Generating outputs...")

	testingSamples := mnist.LoadTestingDataSet()

	mnist.SaveReconstructionGrid("output.png", func(img []float64) []float64 {
		return reconstruct(layers, img)
	}, testingSamples, ReconstructionGridSize, ReconstructionGridSize)
}

func buildLayers() rbm.DBN {
	res := make(rbm.DBN, len(HiddenSizes))
	for i, size := range HiddenSizes {
		inputSize := ImageSize
		if i > 0 {
			inputSize = HiddenSizes[i-1]
		}
		res[i] = rbm.NewRBM(inputSize, size)
	}
	return res
}

func reconstruct(dbn rbm.DBN, img []float64) []float64 {
	binaryInput := make([]bool, len(img))
	for i, x := range img {
		if x > 0.5 {
			binaryInput[i] = true
		}
	}
	output := dbn.Sample(binaryInput)
	binaryInput = dbn.SampleInput(output)
	res := make([]float64, len(binaryInput))
	for i, b := range binaryInput {
		if b {
			res[i] = 1
		}
	}
	return res
}
