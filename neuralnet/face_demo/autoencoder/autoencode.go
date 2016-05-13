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
	FilterSize   = 5
	FilterStride = 3
	FilterCount  = 5
	HiddenSize   = 100
	StepSize     = 0.1
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

	convOutWidth := (width-FilterSize)/FilterStride + 1
	convOutHeight := (height-FilterSize)/FilterStride + 1

	network, _ := neuralnet.NewNetwork([]neuralnet.LayerPrototype{
		&neuralnet.ConvParams{
			Activation:   neuralnet.Sigmoid{},
			FilterCount:  FilterCount,
			FilterWidth:  FilterSize,
			FilterHeight: FilterSize,
			Stride:       FilterStride,
			InputWidth:   width,
			InputHeight:  height,
			InputDepth:   3,
		},
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  convOutWidth * convOutHeight * FilterCount,
			OutputCount: HiddenSize,
		},
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  HiddenSize,
			OutputCount: width * height * 3,
		},
	})

	tensorSlices := make([][]float64, len(tensors))
	for i, tensor := range tensors {
		tensorSlices[i] = tensor.Data
	}

	trainer := neuralnet.SGD{
		CostFunc: neuralnet.MeanSquaredCost{},
		Inputs:   tensorSlices,
		Outputs:  tensorSlices,
		StepSize: StepSize,
		Epochs:   1,
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

	epochNum := 0
TrainLoop:
	for {
		select {
		case <-killChan:
			log.Print("Finishing due to interrupt.")
			break TrainLoop
		default:
		}
		log.Println("Running epoch", epochNum, "error is", trainerError(network, &trainer))
		epochNum++
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
