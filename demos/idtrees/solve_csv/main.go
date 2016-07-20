package main

import (
	"fmt"
	"log"
	"os"

	"github.com/unixpickle/weakai/idtrees"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: idtrees <data.csv>")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "  The first row of the input CSV file specifies field names.")
		fmt.Fprintln(os.Stderr, "  Fields with names starting with _ are ignored.")
		fmt.Fprintln(os.Stderr, "  The field whose name begins with * is identified by the tree.")
		fmt.Fprintln(os.Stderr, "")
		os.Exit(1)
	}

	log.Println("Reading CSV file...")

	f, err := os.Open(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error opening file:", err)
		os.Exit(1)
	}
	defer f.Close()

	samples, keys, err := ReadCSV(f)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	log.Println("Generating tree...")
	tree := idtrees.ID3(samples, keys, 0)
	log.Println("Printing tree...")

	fmt.Println(tree)
}
