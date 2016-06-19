package main

import (
	"errors"
	"image"
	"log"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	HiddenSize = 50
	BatchSize  = 200
	MaxEpochs  = 100000
)

var StepSizes = []float64{1e-3, 1e-4, 1e-5, 1e-6}

func Autoencode(images <-chan image.Image) (neuralnet.Network, error) {
	firstImage := <-images
	if firstImage == nil {
		return nil, errors.New("no readable images")
	}

	width := firstImage.Bounds().Dx()
	height := firstImage.Bounds().Dy()

	log.Print("Reading images...")

	tensors := []*neuralnet.Tensor3{ImageTensor(firstImage)}
	for img := range images {
		if img.Bounds().Dx() != width || img.Bounds().Dy() != height {
			log.Printf("Image size %d,%d does not match %d,%d",
				img.Bounds().Dx(), img.Bounds().Dy(),
				width, height)
		} else {
			tensors = append(tensors, ImageTensor(img))
		}
	}

	log.Print("Training network...")
	log.Print("Press ctrl+c to move on to the next step size.")

	network := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  width * height * 3,
			OutputCount: HiddenSize,
		},
		neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize,
			OutputCount: HiddenSize,
		},
		neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize,
			OutputCount: width * height * 3,
		},
	}
	network.Randomize()

	tensorSlices := make([]linalg.Vector, len(tensors))
	for i, tensor := range tensors {
		tensorSlices[i] = tensor.Data
	}
	samples := neuralnet.VectorSampleSet(tensorSlices, tensorSlices)

	batcher := &neuralnet.BatchRGradienter{
		Learner:  network.BatchLearner(),
		CostFunc: neuralnet.SigmoidCECost{},
	}
	rms := &neuralnet.RMSProp{Gradienter: batcher}

	for i, stepSize := range StepSizes {
		log.Printf("Using step size %f (%d out of %d)", stepSize, i+1, len(StepSizes))
		neuralnet.SGDInteractive(rms, samples, stepSize, BatchSize, func() bool {
			cost := neuralnet.TotalCost(batcher.CostFunc, network, samples)
			log.Println("Current cost is", cost)
			return true
		})
	}

	network = append(network, neuralnet.Sigmoid{})

	return network, nil
}
