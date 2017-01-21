package idtrees

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

// ID3 generates a Tree using the ID3 algorithm.
//
// The maxGos argument specifies the maximum number
// of Goroutines to use during tree generation.
// If maxGos is 0, then GOMAXPROCS is used.
func ID3(samples []Sample, attrs []Attr, maxGos int) *Tree {
	return LimitedID3(samples, attrs, maxGos, -1)
}

// LimitedID3 is like ID3, but it will never produce a
// tree deeper than maxDepth.
// The depth of the tree is counted as the number of
// branches needed to get to a leaf.
// Thus, a tree with no branches has depth 0.
func LimitedID3(samples []Sample, attrs []Attr, maxGos, maxDepth int) *Tree {
	if maxGos == 0 {
		maxGos = runtime.GOMAXPROCS(0)
	}
	baseEntropy := newEntropyCounter(samples).Entropy()
	return id3(samples, attrs, maxGos, maxDepth, baseEntropy)
}

func id3(samples []Sample, attrs []Attr, maxGos, maxDepth int, entropy float64) *Tree {
	if entropy == 0 || maxDepth == 0 {
		return createLeaf(samples)
	}

	attrChan := make(chan Attr, len(attrs))
	for _, a := range attrs {
		attrChan <- a
	}
	close(attrChan)

	splitChan := make(chan *potentialSplit)

	var wg sync.WaitGroup
	for i := 0; i < maxGos; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for attr := range attrChan {
				split := createPotentialSplit(samples, attr)
				if split != nil {
					splitChan <- split
				}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(splitChan)
	}()

	var bestSplit *potentialSplit
	for split := range splitChan {
		if bestSplit == nil || split.Entropy < bestSplit.Entropy {
			bestSplit = split
		}
	}

	if bestSplit == nil || bestSplit.Entropy >= entropy ||
		bestSplit.numBranches() < 2 {
		return createLeaf(samples)
	}

	if bestSplit.Threshold != nil {
		less := id3(bestSplit.NumSplitSamples[0], attrs, maxGos, maxDepth-1,
			bestSplit.NumSplitEntropies[0])
		greater := id3(bestSplit.NumSplitSamples[1], attrs, maxGos, maxDepth-1,
			bestSplit.NumSplitEntropies[1])
		return &Tree{
			Attr: bestSplit.Attr,
			NumSplit: &NumSplit{
				Threshold: bestSplit.Threshold,
				LessEqual: less,
				Greater:   greater,
			},
		}
	}

	res := &Tree{
		Attr:     bestSplit.Attr,
		ValSplit: ValSplit{},
	}
	for class, samples := range bestSplit.ValSplitSamples {
		tree := id3(samples, attrs, maxGos, maxDepth-1, bestSplit.ValSplitEntropies[class])
		res.ValSplit[class] = tree
	}
	return res
}

func createLeaf(samples []Sample) *Tree {
	counts := map[Class]int{}
	for _, s := range samples {
		counts[s.Class()]++
	}
	res := &Tree{Classification: map[Class]float64{}}
	totalScaler := 1 / float64(len(samples))
	for class, count := range counts {
		res.Classification[class] = float64(count) * totalScaler
	}
	return res
}

type potentialSplit struct {
	Attr    Attr
	Entropy float64

	ValSplitEntropies map[Val]float64
	ValSplitSamples   map[Val][]Sample

	Threshold         Val
	NumSplitEntropies [2]float64
	NumSplitSamples   [2][]Sample
}

// numBranches returns the number of non-empty branches
// resulting from the split.
func (p *potentialSplit) numBranches() int {
	var count int
	if p.Threshold != nil {
		if len(p.NumSplitSamples[0]) > 0 {
			count++
		}
		if len(p.NumSplitSamples[1]) > 0 {
			count++
		}
	} else {
		for _, split := range p.ValSplitSamples {
			if len(split) > 0 {
				count++
			}
		}
	}
	return count
}

func createPotentialSplit(samples []Sample, attr Attr) *potentialSplit {
	if len(samples) == 0 {
		panic("cannot split 0 samples")
	}

	val1 := samples[0].Attr(attr)
	switch val1.(type) {
	case int64:
		return createIntSplit(copySampleSlice(samples), attr)
	case float64:
		return createFloatSplit(copySampleSlice(samples), attr)
	}

	res := &potentialSplit{
		Attr:              attr,
		ValSplitEntropies: map[Val]float64{},
		ValSplitSamples:   map[Val][]Sample{},
	}

	for _, s := range samples {
		v := s.Attr(attr)
		res.ValSplitSamples[v] = append(res.ValSplitSamples[v], s)
	}

	totalDivider := 1 / float64(len(samples))
	for attrVal, s := range res.ValSplitSamples {
		e := newEntropyCounter(s).Entropy()
		res.ValSplitEntropies[attrVal] = e
		res.Entropy += float64(len(s)) * totalDivider * e
	}

	return res
}

func createIntSplit(samples []Sample, attr Attr) *potentialSplit {
	sorter := &intSorter{
		sampleSorter: sampleSorter{
			Attr:    attr,
			Samples: samples,
		},
	}
	sort.Sort(sorter)

	lastValue := sorter.Samples[0].Attr(attr).(int64)
	var cutoffIdxs []int
	var cutoffs []Val
	for i := 1; i < len(sorter.Samples); i++ {
		val := sorter.Samples[i].Attr(attr).(int64)
		if val > lastValue {
			cutoffIdxs = append(cutoffIdxs, i)
			cutoffs = append(cutoffs, lastValue+(val-lastValue)/2)
			lastValue = val
		}
	}

	return createNumericSplit(sorter.sampleSorter, cutoffIdxs, cutoffs)
}

func createFloatSplit(samples []Sample, attr Attr) *potentialSplit {
	sorter := &floatSorter{
		sampleSorter: sampleSorter{
			Attr:    attr,
			Samples: samples,
		},
	}
	sort.Sort(sorter)

	lastValue := sorter.Samples[0].Attr(attr).(float64)
	var cutoffIdxs []int
	var cutoffs []Val
	for i := 1; i < len(sorter.Samples); i++ {
		val := sorter.Samples[i].Attr(attr).(float64)
		if val > lastValue {
			cutoffIdxs = append(cutoffIdxs, i)
			cutoffs = append(cutoffs, lastValue+(val-lastValue)/2)
			lastValue = val
		}
	}

	return createNumericSplit(sorter.sampleSorter, cutoffIdxs, cutoffs)
}

func createNumericSplit(s sampleSorter, cutoffIdxs []int, cutoffs []Val) *potentialSplit {
	if len(cutoffIdxs) == 0 {
		return nil
	}

	best := &potentialSplit{
		Attr: s.Attr,
	}

	lessEntropy := newEntropyCounter(s.Samples[:cutoffIdxs[0]])
	greaterEntropy := newEntropyCounter(s.Samples[cutoffIdxs[0]:])

	countDivider := 1 / float64(len(s.Samples))
	for i, cutoffIdx := range cutoffIdxs {
		if i != 0 {
			lastIdx := cutoffIdxs[i-1]
			for j := lastIdx; j < cutoffIdx; j++ {
				lessEntropy.Add(s.Samples[j])
				greaterEntropy.Remove(s.Samples[j])
			}
		}
		lessE := lessEntropy.Entropy()
		greaterE := greaterEntropy.Entropy()
		entropy := countDivider * (float64(lessEntropy.totalCount)*lessE +
			float64(greaterEntropy.totalCount)*greaterE)
		if entropy < best.Entropy || i == 0 {
			best.Entropy = entropy
			best.NumSplitEntropies[0] = lessE
			best.NumSplitEntropies[1] = greaterE
			best.NumSplitSamples[0] = s.Samples[:cutoffIdx]
			best.NumSplitSamples[1] = s.Samples[cutoffIdx:]
			best.Threshold = cutoffs[i]
		}
	}

	return best
}

type entropyCounter struct {
	classCounts map[Class]int
	totalCount  int
}

func newEntropyCounter(s []Sample) *entropyCounter {
	res := &entropyCounter{
		classCounts: map[Class]int{},
		totalCount:  len(s),
	}
	for _, sample := range s {
		res.classCounts[sample.Class()]++
	}
	return res
}

func (e *entropyCounter) Entropy() float64 {
	var entropy float64
	countScaler := 1 / float64(e.totalCount)
	for _, count := range e.classCounts {
		if count == 0 {
			continue
		}
		probability := float64(count) * countScaler
		entropy -= probability * math.Log(probability)
	}
	return entropy
}

func (e *entropyCounter) Add(s Sample) {
	e.classCounts[s.Class()]++
	e.totalCount++
}

func (e *entropyCounter) Remove(s Sample) {
	e.classCounts[s.Class()]--
	e.totalCount--
}

func copySampleSlice(s []Sample) []Sample {
	res := make([]Sample, len(s))
	copy(res, s)
	return res
}

type sampleSorter struct {
	Attr    Attr
	Samples []Sample
}

func (s *sampleSorter) Len() int {
	return len(s.Samples)
}

func (s *sampleSorter) Swap(i, j int) {
	s.Samples[i], s.Samples[j] = s.Samples[j], s.Samples[i]
}

type intSorter struct {
	sampleSorter
}

func (i *intSorter) Less(k, j int) bool {
	kVal := i.Samples[k].Attr(i.Attr).(int64)
	jVal := i.Samples[j].Attr(i.Attr).(int64)
	return kVal < jVal
}

type floatSorter struct {
	sampleSorter
}

func (f *floatSorter) Less(k, j int) bool {
	kVal := f.Samples[k].Attr(f.Attr).(float64)
	jVal := f.Samples[j].Attr(f.Attr).(float64)
	return kVal < jVal
}
