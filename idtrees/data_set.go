package idtrees

import "math"

type DataSet struct {
	Entries []Entry
	Fields  []Field
}

func (d *DataSet) classes() []Value {
	res := []Value{}
	seen := map[Value]bool{}
	for _, e := range d.Entries {
		if !seen[e.Class()] {
			seen[e.Class()] = true
			res = append(res, e.Class())
		}
	}
	return res
}

func (d *DataSet) dominantClass() Value {
	counts := map[Value]int{}
	for _, e := range d.Entries {
		counts[e.Class()]++
	}
	var bestValue Value
	var bestCount int
	for val, count := range counts {
		if count > bestCount {
			bestValue = val
			bestCount = count
		}
	}
	return bestValue
}

func (d *DataSet) filter(fieldIndex int, v Value) *DataSet {
	res := &DataSet{Fields: d.Fields, Entries: []Entry{}}
	for _, entry := range d.Entries {
		if entry.FieldValues()[fieldIndex] == v {
			res.Entries = append(res.Entries, entry)
		}
	}
	return res
}

func (d *DataSet) statsForFilter(fieldIndex int, v Value) (disorder float64, count int) {
	classDistribution := map[Value]int{}
	for _, entry := range d.Entries {
		if entry.FieldValues()[fieldIndex] == v {
			class := entry.Class()
			classDistribution[class]++
			count++
		}
	}

	if len(classDistribution) <= 1 {
		return 0, count
	}

	// See https://en.wikipedia.org/wiki/Entropy_%28information_theory%29#Characterization
	for _, amount := range classDistribution {
		fraction := float64(amount) / float64(count)
		if fraction != 0 {
			disorder -= fraction * math.Log(fraction)
		}
	}

	return
}
