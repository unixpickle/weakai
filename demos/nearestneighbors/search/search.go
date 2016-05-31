package main

type searchSorter struct {
	Search  []float64
	Entries []normalizedIndexEntry
}

func (s searchSorter) Len() int {
	return len(s.Entries)
}

func (s searchSorter) Less(i, j int) bool {
	return s.entryRelevance(i) > s.entryRelevance(j)
}

func (s searchSorter) Swap(i, j int) {
	s.Entries[i], s.Entries[j] = s.Entries[j], s.Entries[i]
}

func (s searchSorter) entryRelevance(i int) float64 {
	entry := s.Entries[i]
	return entry.relevanceForQuery(s.Search)
}
