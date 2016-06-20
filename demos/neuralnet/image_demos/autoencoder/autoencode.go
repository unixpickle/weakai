package main

import (
	"errors"
	"image"
	"log"
	"math"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	HiddenSize1 = 300
	HiddenSize2 = 100
	BatchSize   = 200
	MaxEpochs   = 100000
)

var StepSizes = []float64{3e-3, 1e-3, 1e-4}

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

	tensorSlices := make([]linalg.Vector, len(tensors))
	for i, tensor := range tensors {
		tensorSlices[i] = tensor.Data
	}
	samples := neuralnet.VectorSampleSet(tensorSlices, tensorSlices)

	average, stddev := statisticalInfo(tensorSlices)

	network := neuralnet.Network{
		&neuralnet.RescaleLayer{
			Bias:  -average,
			Scale: 1 / stddev,
		},
		&neuralnet.DenseLayer{
			InputCount:  width * height * 3,
			OutputCount: HiddenSize1,
		},
		neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize1,
			OutputCount: HiddenSize2,
		},
		neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize2,
			OutputCount: HiddenSize1,
		},
		neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize1,
			OutputCount: width * height * 3,
		},
	}
	network.Randomize()

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

func statisticalInfo(samples []linalg.Vector) (mean, stddev float64) {
	var count int
	for _, v := range samples {
		for _, x := range v {
			mean += x
			count++
		}
	}

	mean /= float64(count)

	for _, v := range samples {
		for _, x := range v {
			stddev += math.Pow(x-mean, 2)
			count++
		}
	}
	stddev /= float64(count)
	stddev = math.Sqrt(stddev)

	return
}
