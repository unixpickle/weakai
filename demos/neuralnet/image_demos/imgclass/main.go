package main

import (
	"fmt"
	"os"

	_ "image/jpeg"
	_ "image/png"
)

func main() {
	if len(os.Args) < 2 {
		dieUsage()
	}
	switch os.Args[1] {
	case "train":
		TrainCmd(os.Args[2], os.Args[3])
	case "classify":
		ClassifyCmd(os.Args[2], os.Args[3])
	case "dream":
		DreamCmd(os.Args[2], os.Args[3])
	default:
		dieUsage()
	}
}

func dieUsage() {
	fmt.Fprintln(os.Stderr, "Usage: imgclass train <network_file> <image_dir>\n"+
		"                classify <network_file> <image>\n"+
		"                dream <network_file> <image-out>\n\n"+
		"The image directory should have a sub-directory per class.")
	os.Exit(1)
}
