package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage: search <index.json> <query>")
		os.Exit(1)
	}

	index, err := ReadIndex(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	results := index.SearchResults(os.Args[2])
	if len(results) == 0 {
		fmt.Fprintln(os.Stderr, "No results were relevant to your search.")
		os.Exit(1)
	}
	for i := 0; i < len(results) && i < 10; i++ {
		fmt.Println(results[i])
	}
}
