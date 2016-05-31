package main

import (
	"errors"
	"image"
	"log"
	"math"

	"github.com/unixpickle/weakai/neuralnet"
)

const (
	FilterSize   = 4
	FilterCount  = 4
	FilterStride = 2
	StepSize     = 1e-6
	MaxEpochs    = 100000
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

	borderTargetWidth := (width/FilterStride + FilterSize) - 1
	borderTargetHeight := (height/FilterStride + FilterSize) - 1
	widthPadding := borderTargetWidth - convOutWidth
	heightPadding := borderTargetHeight - convOutHeight

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
		&neuralnet.BorderParams{
			InputWidth:   convOutWidth,
			InputHeight:  convOutHeight,
			InputDepth:   FilterCount,
			LeftBorder:   widthPadding / 2,
			RightBorder:  widthPadding - (widthPadding / 2),
			TopBorder:    heightPadding / 2,
			BottomBorder: heightPadding - (heightPadding / 2),
		},
		&neuralnet.ConvGrowParams{
			ConvParams: neuralnet.ConvParams{
				Activation:   neuralnet.Sigmoid{},
				FilterCount:  3,
				FilterWidth:  FilterSize,
				FilterHeight: FilterSize,
				Stride:       1,
				InputWidth:   borderTargetWidth,
				InputHeight:  borderTargetHeight,
				InputDepth:   FilterCount,
			},
			InverseStride: FilterStride,
		},
		&neuralnet.BorderParams{
			InputWidth:  width - (width % FilterStride),
			InputHeight: height - (height % FilterStride),
			InputDepth:  3,
			LeftBorder:  width % FilterStride,
			TopBorder:   height % FilterStride,
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
		Epochs:   100000,
	}

	network.Randomize()
	trainer.TrainInteractive(network)

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
