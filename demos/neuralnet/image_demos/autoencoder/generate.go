package main

import (
	"fmt"
	"io/ioutil"
	"os"
)

func Generate() {
	imageDir := os.Args[2]
	outputFile := os.Args[3]

	images, err := ReadImages(imageDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	encoder, err := Autoencode(images)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Could not autoencode:", err)
		os.Exit(1)
	}

	serialized, err := encoder.Serialize()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Could not serialize:", err)
		os.Exit(1)
	}

	if err := ioutil.WriteFile(outputFile, serialized, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
