package main

import (
	"fmt"
	_ "image/jpeg"
	_ "image/png"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: autoencoder gen <image-dir> <output-file>")
		fmt.Fprintln(os.Stderr, "       autoencoder run <autoencoder> <image-in> <image-out>")
		os.Exit(1)
	}

	if os.Args[1] == "gen" {
		Generate()
	} else if os.Args[1] == "run" {
		Run()
	} else {
		fmt.Fprintln(os.Stderr, "Unknown sub-command:", os.Args[1])
		os.Exit(1)
	}
}
