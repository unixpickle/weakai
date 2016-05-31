package main

import (
	"encoding/json"
	"io/ioutil"
	"math"
	"sort"
	"strings"
)

type rawIndexEntry struct {
	URL      string         `json:"url"`
	Keywords map[string]int `json:"keywords"`
}

type normalizedIndexEntry struct {
	URL    string
	Vector []float64
}

func (n normalizedIndexEntry) relevanceForQuery(vector []float64) float64 {
	var dotProduct float64
	for i, x := range vector {
		dotProduct += x * n.Vector[i]
	}
	return dotProduct
}

type Index struct {
	entries  []normalizedIndexEntry
	keywords []string
}

func ReadIndex(path string) (idx *Index, err error) {
	contents, err := ioutil.ReadFile(path)
	if err != nil {
		return
	}

	var rawEntries []rawIndexEntry
	if err := json.Unmarshal(contents, &rawEntries); err != nil {
		return nil, err
	}

	keywords := keywordsFromEntries(rawEntries)
	idx = &Index{entries: []normalizedIndexEntry{}, keywords: keywords}

	for _, raw := range rawEntries {
		idx.addRawEntry(raw)
	}

	return idx, nil
}

func (i *Index) SearchResults(query string) []string {
	fields := strings.Fields(strings.ToLower(query))
	wordFrequency := map[string]int{}
	for _, f := range fields {
		wordFrequency[f] = wordFrequency[f] + 1
	}

	vector := i.vectorForKeywordMap(wordFrequency)
	if vector == nil {
		// None of the words in the user's search were indexed keywords.
		return []string{}
	}

	entries := make([]normalizedIndexEntry, len(i.entries))
	copy(entries, i.entries)
	sorter := searchSorter{Search: vector, Entries: entries}
	sort.Sort(sorter)

	res := make([]string, 0, len(sorter.Entries))
	for _, entry := range sorter.Entries {
		if entry.relevanceForQuery(vector) > 0 {
			res = append(res, entry.URL)
		}
	}
	return res
}

func (i *Index) vectorForKeywordMap(m map[string]int) []float64 {
	vector := make([]float64, len(i.keywords))
	var magnitude float64
	for keyword, count := range m {
		idx := sort.SearchStrings(i.keywords, keyword)
		if idx == len(i.keywords) || i.keywords[idx] != keyword {
			continue
		}
		vector[idx] += float64(count)
		magnitude += float64(count)
	}
	magnitude = math.Sqrt(magnitude)
	if magnitude == 0 {
		return nil
	}
	for j, val := range vector {
		vector[j] = val / magnitude
	}
	return vector
}

func (i *Index) addRawEntry(raw rawIndexEntry) {
	vec := i.vectorForKeywordMap(raw.Keywords)
	if vec == nil {
		return
	}
	i.entries = append(i.entries, normalizedIndexEntry{URL: raw.URL, Vector: vec})
}

func keywordsFromEntries(entries []rawIndexEntry) []string {
	words := map[string]bool{}
	for _, entry := range entries {
		for word := range entry.Keywords {
			words[word] = true
		}
	}

	res := make([]string, 0, len(words))
	for word := range words {
		res = append(res, word)
	}

	sort.Strings(res)

	return res
}
