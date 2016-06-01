package main

import (
	"fmt"
	"image"
	"image/png"
	"io/ioutil"
	"os"

	"github.com/unixpickle/weakai/neuralnet"
)

func Run() {
	encoderPath := os.Args[2]

	encoderData, err := ioutil.ReadFile(encoderPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	network, err := neuralnet.DeserializeNetwork(encoderData)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	inputPath := os.Args[3]
	outputPath := os.Args[4]

	f, err := os.Open(inputPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	inputImage, _, err := image.Decode(f)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	if !network.SetInput(ImageTensor(inputImage).Data) {
		fmt.Fprintln(os.Stderr, "Bad image dimensions.")
		os.Exit(1)
	}

	network.PropagateForward()

	tensor := &neuralnet.Tensor3{
		Width:  inputImage.Bounds().Dx(),
		Height: inputImage.Bounds().Dy(),
		Depth:  3,
	}
	tensor.Data = network.Output()

	image := ImageFromTensor(tensor)
	outFile, err := os.Create(outputPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer outFile.Close()
	if err := png.Encode(outFile, image); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
