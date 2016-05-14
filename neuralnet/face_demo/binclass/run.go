package main

import (
	"errors"
	"fmt"
	"image"
	"io/ioutil"
	"os"

	"github.com/unixpickle/weakai/neuralnet"
)

func Run(classifier, sample string) error {
	contents, err := ioutil.ReadFile(classifier)
	if err != nil {
		return err
	}
	network, err := neuralnet.DeserializeNetwork(contents)
	if err != nil {
		return errors.New("Couldn't deserialize network: " + err.Error())
	}

	imageFile, err := os.Open(sample)
	if err != nil {
		return err
	}

	defer imageFile.Close()

	img, _, err := image.Decode(imageFile)
	if err != nil {
		return err
	}

	tensor := ImageTensor(img)
	network.SetInput(tensor.Data)
	network.PropagateForward()
	if network.Output()[0] > network.Output()[1] {
		fmt.Println("Class 0")
	} else {
		fmt.Println("Class 1")
	}
	fmt.Println("Scores were", network.Output()[0], network.Output()[1])

	return nil
}
