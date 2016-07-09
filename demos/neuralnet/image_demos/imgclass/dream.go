package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"log"
	"math/rand"
	"os"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

func DreamCmd(netPath, imgPath string) {
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

	convIn := network[1].(*neuralnet.ConvLayer)
	inputImage := &autofunc.Variable{
		Vector: make(linalg.Vector, convIn.InputWidth*convIn.InputHeight*
			convIn.InputDepth),
	}
	for i := range inputImage.Vector {
		inputImage.Vector[i] = rand.Float64()*0.01 + 0.5
	}

	desiredOut := linalg.Vector{0, 1}
	cost := neuralnet.DotCost{}
	grad := autofunc.NewGradient([]*autofunc.Variable{inputImage})
	for i := 0; i < 1000; i++ {
		output := network.Apply(inputImage)
		costOut := cost.Cost(desiredOut, output)
		grad.Zero()
		log.Println("cost is", costOut.Output()[0])
		costOut.PropagateGradient(linalg.Vector{1}, grad)
		grad.AddToVars(-0.01)
	}

	newImage := image.NewRGBA(image.Rect(0, 0, convIn.InputWidth, convIn.InputHeight))
	var idx int
	for y := 0; y < convIn.InputHeight; y++ {
		for x := 0; x < convIn.InputWidth; x++ {
			r := uint8(0xff * inputImage.Vector[idx])
			g := uint8(0xff * inputImage.Vector[idx+1])
			b := uint8(0xff * inputImage.Vector[idx+2])
			newImage.SetRGBA(x, y, color.RGBA{
				R: r,
				G: g,
				B: b,
				A: 0xff,
			})
			idx += 3
		}
	}

	output, err := os.Create(imgPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to create output file:", err)
		os.Exit(1)
	}
	defer output.Close()
	png.Encode(output, newImage)
}
