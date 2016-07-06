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

	img, err := ReadImageFile(imgPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error reading image:", err)
		os.Exit(1)
	}

	output := network.Apply(&autofunc.Variable{Vector: img}).Output()

	for i, x := range output {
		fmt.Printf("Class %d: probability %f\n", i, math.Exp(x))
	}
}
