package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

type IndexEntry struct {
	URL      string         `json:"url"`
	Keywords map[string]int `json:"keywords"`
}

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "Usage: prune_index <input.json> <output.json> <threshold>")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "The threshold argument is a number between 0 and 1.")
		fmt.Fprintln(os.Stderr, "A value of 0 will remove all keywords, 1 will keep all keywords.")
		os.Exit(1)
	}

	threshold, err := strconv.ParseFloat(os.Args[3], 64)
	if err != nil || threshold < 0 || threshold > 1 {
		fmt.Fprintln(os.Stderr, "Invalid threshold:", os.Args[3])
		os.Exit(1)
	}

	var entries []IndexEntry
	contents, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := json.Unmarshal(contents, &entries); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	ubiquity := keywordUbiquity(entries)
	keepKeywords := map[string]bool{}
	pruned := []string{}
	for word, ubiquity := range ubiquity {
		if ubiquity < threshold {
			keepKeywords[word] = true
		} else {
			pruned = append(pruned, word)
		}
	}

	fmt.Println("Pruned", len(pruned), "words:", strings.Join(pruned, ", "))

	filterKeywords(entries, keepKeywords)
	data, _ := json.Marshal(entries)
	if err := ioutil.WriteFile(os.Args[2], data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func keywordUbiquity(entries []IndexEntry) map[string]float64 {
	ubiquity := map[string]float64{}
	ubiquityPerPage := 1 / float64(len(entries))
	for _, entry := range entries {
		for word := range entry.Keywords {
			ubiquity[word] += ubiquityPerPage
		}
	}
	return ubiquity
}

func filterKeywords(entries []IndexEntry, filter map[string]bool) {
	for i, entry := range entries {
		newKeywords := map[string]int{}
		for word, count := range entry.Keywords {
			if filter[word] {
				newKeywords[word] = count
			}
		}
		entries[i] = IndexEntry{URL: entry.URL, Keywords: newKeywords}
	}
}
