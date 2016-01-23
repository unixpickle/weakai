package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "Usage: indexer <wiki_url> <page_count> <output.json>")
		os.Exit(1)
	}

	pageCount, err := strconv.Atoi(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Invalid page count:", os.Args[3])
		os.Exit(1)
	}

	res, err := IndexWikipedia(pageCount, os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	encoded, _ := json.Marshal(res)
	if err := ioutil.WriteFile(os.Args[3], encoded, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
