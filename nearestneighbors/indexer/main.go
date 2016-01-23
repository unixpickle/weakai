package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

func main() {
	if len(os.Args) != 5 {
		fmt.Fprintln(os.Stderr, "Usage: indexer <keywords.txt> <wiki_url> <page_count> <output.json>")
		os.Exit(1)
	}

	pageCount, err := strconv.Atoi(os.Args[3])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Invalid page count:", os.Args[3])
		os.Exit(1)
	}

	keywords, err := readKeywords()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	res, err := IndexWikipedia(keywords, pageCount, os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	encoded, _ := json.Marshal(res)
	if err := ioutil.WriteFile(os.Args[4], encoded, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func readKeywords() ([]string, error) {
	contents, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(contents), "\n")
	res := make([]string, 0, len(lines))
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if len(trimmed) == 0 || trimmed[0] == '#' {
			continue
		}
		res = append(res, strings.ToLower(trimmed))
	}
	return res, nil
}
