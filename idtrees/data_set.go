package idtrees

import "math"

type DataSet []Entry

func (d DataSet) allFields() []Field {
	if len(d) == 0 {
		return []Field{}
	}
	res := make([]Field, 0, len(d[0].FieldValues()))
	for field := range d[0].FieldValues() {
		res = append(res, field)
	}
	return res
}

func (d DataSet) classes() []Value {
	res := []Value{}
	seen := map[Value]bool{}
	for _, e := range d {
		if !seen[e.Class()] {
			seen[e.Class()] = true
			res = append(res, e.Class())
		}
	}
	return res
}

func (d DataSet) filter(f Field, v Value) DataSet {
	res := DataSet{}
	for _, entry := range d {
		if entry.FieldValues()[f] == v {
			res = append(res, entry)
		}
	}
	return res
}

func (d DataSet) disorder() float64 {
	if len(d) == 0 {
		return 0
	}

	classes := d.classes()
	if len(classes) <= 1 {
		return 0
	}

	classDistribution := map[Value]int{}
	for _, entry := range d {
		classDistribution[entry.Class()]++
	}

	// See https://en.wikipedia.org/wiki/Entropy_%28information_theory%29#Characterization
	var res float64
	for _, class := range classes {
		fraction := float64(classDistribution[class]) / float64(len(d))
		if fraction != 0 {
			res -= fraction * math.Log(fraction)
		}
	}

	return res
}
