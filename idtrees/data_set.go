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

func (d *DataSet) filter(fieldIndex int, v Value) *DataSet {
	res := &DataSet{Fields: d.Fields, Entries: []Entry{}}
	for _, entry := range d.Entries {
		if entry.FieldValues()[fieldIndex] == v {
			res.Entries = append(res.Entries, entry)
		}
	}
	return res
}

func (d *DataSet) disorder() float64 {
	if len(d.Entries) == 0 {
		return 0
	}

	classDistribution := map[Value]int{}
	for _, entry := range d.Entries {
		classDistribution[entry.Class()]++
	}

	if len(classDistribution) == 1 {
		return 0
	}

	// See https://en.wikipedia.org/wiki/Entropy_%28information_theory%29#Characterization
	var res float64
	for _, amount := range classDistribution {
		fraction := float64(amount) / float64(len(d.Entries))
		if fraction != 0 {
			res -= fraction * math.Log(fraction)
		}
	}

	return res
}
