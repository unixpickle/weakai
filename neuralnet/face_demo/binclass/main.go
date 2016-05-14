package main

import (
	"fmt"
	"os"
)

func main() {
	var err error
	if len(os.Args) == 5 && os.Args[1] == "train" {
		err = Train(os.Args[2], os.Args[3], os.Args[4])
	} else if len(os.Args) == 3 && os.Args[1] == "run" {
		err = Run(os.Args[2], os.Args[3])
	} else {
		fmt.Fprintln(os.Stderr, "Usage: binclass train <samples0> <samples1> <classifier-out>\n"+
			"       binclass run <classifier> <sample>")
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
