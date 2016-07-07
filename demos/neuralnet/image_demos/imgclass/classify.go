package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"os"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/weakai/neuralnet"
)

func ClassifyCmd(netPath, imgPath string) {
	networkData, err := ioutil.ReadFile(netPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error reading network:", err)
		os.Exit(1)
	}
	network, err := neuralnet.DeserializeNetwork(networkData)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error deserializing network:", err)
		os.Exit(1)
	}

	img, width, height, err := ReadImageFile(imgPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error reading image:", err)
		os.Exit(1)
	}

	firstLayer := network[1].(*neuralnet.ConvLayer)
	if width != firstLayer.InputWidth || height != firstLayer.InputHeight {
		fmt.Fprintf(os.Stderr, "Expected dimensions %dx%d but got %dx%d\n",
			firstLayer.InputWidth, firstLayer.InputHeight, width, height)
	}

	output := network.Apply(&autofunc.Variable{Vector: img}).Output()

	for i, x := range output {
		fmt.Printf("Class %d: probability %f\n", i, math.Exp(x))
	}
}
