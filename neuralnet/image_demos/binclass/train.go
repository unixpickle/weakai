package main

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io/ioutil"
	"log"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/unixpickle/num-analysis/kahan"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	Filter1Size   = 3
	Filter1Stride = 2
	Filter1Depth  = 10

	Filter2Size   = 10
	Filter2Stride = 4
	Filter2Depth  = 15

	StepSize = 3e-2
)

func Train(samples0, samples1, classifierPath string) error {
	network, err := trainNetwork(samples0, samples1)
	if err != nil {
		return err
	}
	encoded := network.Serialize()
	return ioutil.WriteFile(classifierPath, encoded, 0755)
}

func trainNetwork(samples0, samples1 string) (*neuralnet.Network, error) {
	log.Println("Reading samples...")
	img1, img2, err := readSamples(samples0, samples1)
	if err != nil {
		return nil, err
	}

	w, h, d := img1[0].Width, img1[0].Height, img1[0].Depth

	network, err := buildNetwork(w, h, d)
	if err != nil {
		return nil, err
	}

	trainer := buildTrainer(img1, img2)
	network.Randomize()

	log.Println("Training network (ctrl+c to finish)...")

	log.Printf("Step 0 error %f", totalError(trainer, network))

	killChan := make(chan struct{})

	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt)
		<-c
		signal.Stop(c)
		fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
		close(killChan)
	}()

	stepIdx := 0
	for {
		select {
		case <-killChan:
			log.Println("Stopping due to interrupt")
			return network, nil
		default:
		}
		stepIdx++
		trainer.Train(network)
		log.Printf("Step %d error %f", stepIdx, totalError(trainer, network))
	}
}

func readSamples(dir1, dir2 string) (img1, img2 []*neuralnet.Tensor3, err error) {
	img1, err = readImages(dir1)
	if err != nil {
		return
	}
	img2, err = readImages(dir2)
	if err != nil {
		return
	}

	if len(img1) == 0 || len(img2) == 0 {
		return nil, nil, errors.New("cannot work with empty sample directory")
	}

	w, h := img1[0].Width, img1[0].Height

	for _, tensors := range [][]*neuralnet.Tensor3{img1, img2} {
		for _, tensor := range tensors {
			if tensor.Width != w || tensor.Height != h {
				return nil, nil, errors.New("bad image dimensions")
			}
		}
	}

	return img1, img2, nil
}

func buildNetwork(width, height, depth int) (*neuralnet.Network, error) {
	filter1OutWidth := (width-Filter1Size)/Filter1Stride + 1
	filter1OutHeight := (height-Filter1Size)/Filter1Stride + 1
	filter2OutWidth := (filter1OutWidth-Filter2Size)/Filter2Stride + 1
	filter2OutHeight := (filter1OutHeight-Filter2Size)/Filter2Stride + 1

	return neuralnet.NewNetwork([]neuralnet.LayerPrototype{
		&neuralnet.ConvParams{
			FilterCount:  Filter1Depth,
			FilterWidth:  Filter1Size,
			FilterHeight: Filter1Size,
			Stride:       Filter1Stride,
			InputWidth:   width,
			InputHeight:  height,
			InputDepth:   depth,
			Activation:   neuralnet.Sigmoid{},
		},
		&neuralnet.ConvParams{
			FilterCount:  Filter2Depth,
			FilterWidth:  Filter2Size,
			FilterHeight: Filter2Size,
			Stride:       Filter2Stride,
			InputWidth:   filter1OutWidth,
			InputHeight:  filter1OutHeight,
			InputDepth:   Filter1Depth,
			Activation:   neuralnet.Sigmoid{},
		},
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  filter2OutWidth * filter2OutHeight * Filter2Depth,
			OutputCount: 2,
		},
	})
}

func buildTrainer(img1, img2 []*neuralnet.Tensor3) *neuralnet.SGD {
	input := make([][]float64, 0, len(img1)+len(img2))
	output := make([][]float64, 0, len(img1)+len(img2))

	for i, list := range [][]*neuralnet.Tensor3{img1, img2} {
		for _, sample := range list {
			input = append(input, sample.Data)
			o := make([]float64, 2)
			o[i] = 1
			output = append(output, o)
		}
	}

	return &neuralnet.SGD{
		CostFunc: neuralnet.MeanSquaredCost{},
		Inputs:   input,
		Outputs:  output,
		StepSize: StepSize / float64(len(input)),
		Epochs:   1,
	}
}

func totalError(trainer *neuralnet.SGD, network *neuralnet.Network) float64 {
	sum := kahan.NewSummer64()
	for i, x := range trainer.Inputs {
		network.SetInput(x)
		network.PropagateForward()

		actual := network.Output()
		for j, exp := range trainer.Outputs[i] {
			sum.Add(math.Pow(actual[j]-exp, 2))
		}
	}
	return sum.Sum() / 2
}

func readImages(path string) ([]*neuralnet.Tensor3, error) {
	dir, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer dir.Close()
	contents, err := dir.Readdirnames(0)
	if err != nil {
		return nil, err
	}

	res := make([]*neuralnet.Tensor3, 0, len(contents))
	for _, name := range contents {
		if strings.HasPrefix(name, ".") {
			continue
		}
		f, err := os.Open(filepath.Join(path, name))
		if err != nil {
			return nil, err
		}
		image, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			log.Println("Error reading file:", filepath.Join(path, name))
		} else {
			res = append(res, ImageTensor(image))
		}
	}

	return res, nil
}
