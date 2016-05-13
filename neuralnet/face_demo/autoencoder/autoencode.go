package main

import (
	"errors"
	"fmt"
	"image"
	"log"
	"math"
	"os"
	"os/signal"

	"github.com/unixpickle/weakai/neuralnet"
)

const (
	HiddenSize1      = 100
	HiddenSize2      = 100
	StepSize         = 0.1
	StepDecreaseRate = 1e-3
	Epochs           = 1000
)

func Autoencode(images <-chan image.Image) (*neuralnet.Network, error) {
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
	log.Print("Press ctrl+c to stop on next iteration.")

	network, _ := neuralnet.NewNetwork([]neuralnet.LayerPrototype{
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  width * height * 3,
			OutputCount: HiddenSize1,
		},
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  HiddenSize1,
			OutputCount: HiddenSize2,
		},
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  HiddenSize2,
			OutputCount: width * height * 3,
		},
	})

	tensorSlices := make([][]float64, len(tensors))
	for i, tensor := range tensors {
		tensorSlices[i] = tensor.Data
	}

	trainer := neuralnet.SGD{
		CostFunc:         neuralnet.MeanSquaredCost{},
		Inputs:           tensorSlices,
		Outputs:          tensorSlices,
		StepSize:         StepSize,
		StepDecreaseRate: StepDecreaseRate,
		Epochs:           1,
	}

	network.Randomize()

	killChan := make(chan struct{})

	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt)
		<-c
		signal.Stop(c)
		fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
		close(killChan)
	}()

TrainLoop:
	for i := 0; i < Epochs; i++ {
		select {
		case <-killChan:
			log.Print("Finishing due to interrupt.")
			break TrainLoop
		default:
		}
		log.Println("Running epoch", i, "error is", trainerError(network, &trainer))
		trainer.Train(network)
	}

	return network, nil
}

func trainerError(n *neuralnet.Network, t *neuralnet.SGD) float64 {
	var e float64
	for _, sample := range t.Inputs {
		n.SetInput(sample)
		n.PropagateForward()
		for i, x := range n.Output() {
			e += math.Pow(x-sample[i], 2)
		}
	}
	return e
}
