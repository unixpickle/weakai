package main

import (
	"errors"
	"image"
	"log"
	"math"

	"github.com/unixpickle/hessfree"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	HiddenSize1 = 300
	HiddenSize2 = 100
	MaxSubBatch = 20
)

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

	log.Print("Training network (ctrl+c to finish)...")

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

	ui := hessfree.NewConsoleUI()
	learner := &hessfree.DampingLearner{
		WrappedLearner: &hessfree.NeuralNetLearner{
			Layers:         network,
			Output:         nil,
			Cost:           neuralnet.SigmoidCECost{},
			MaxSubBatch:    MaxSubBatch,
			MaxConcurrency: 2,
		},
		DampingCoeff: 5,
		UI:           ui,
	}
	trainer := hessfree.Trainer{
		Learner:   learner,
		Samples:   samples,
		BatchSize: samples.Len(),
		UI:        ui,
	}
	trainer.Train()

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
