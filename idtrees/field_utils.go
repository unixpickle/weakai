package idtrees

import (
	"fmt"
	"sort"
)

type BoolFieldGetter func(e Entry) bool
type IntFieldGetter func(e Entry) int

// A BoolField is a field which reports BoolValue(true) and
// BoolValue(false) as its potential values.
type BoolField string

// String returns string(b)
func (b BoolField) String() string {
	return string(b)
}

// Values returns BoolValue(true) and BoolValue(false).
func (b BoolField) Values() []Value {
	return []Value{BoolValue(true), BoolValue(false)}
}

// A BoolValue is an fmt.Stringer that returns "true" or "false"
// depending on its underlying value.
type BoolValue bool

// String returns "true" if b and "false" otherwise.
func (b BoolValue) String() string {
	if b {
		return "true"
	} else {
		return "false"
	}
}

// A StringValue is an fmt.Stringer that returns itself.
type StringValue string

// String returns string(s).
func (s StringValue) String() string {
	return string(s)
}

// CreateBoolField generates a field with two possibile answers.
// It adds the field to each Entry in the DataSet, specifying the
// answer returned by g.
// It returns the generated Field.
func CreateBoolField(d DataSet, g BoolFieldGetter, label string) BoolField {
	f := BoolField(label)
	for _, entry := range d {
		entry.FieldValues()[f] = BoolValue(g(entry))
	}
	return f
}

// CreateBisectingIntFields generates zero or more fields which
// divide up integer values associated with each Entry.
// It returns the generated Fields, of which there may be none.
//
// The labelFmt argument should be a format string with a "%d" in it.
// This will be used to generate labels for each of the fields.
// It should look something like "X is greater than %d.".
func CreateBisectingIntFields(d DataSet, g IntFieldGetter, labelFmt string) []BoolField {
	possibilities := []int{}
	seenPossibilities := map[int]bool{}
	for _, entry := range d {
		val := g(entry)
		if !seenPossibilities[val] {
			seenPossibilities[val] = true
			possibilities = append(possibilities, val)
		}
	}
	sort.Ints(possibilities)

	if len(possibilities) <= 1 {
		return []BoolField{}
	}

	res := make([]BoolField, 0, len(possibilities)-1)
	for i := 0; i < len(possibilities)-1; i++ {
		middle := (possibilities[i] + possibilities[i+1]) / 2
		field := BoolField(fmt.Sprintf(labelFmt, middle))
		res = append(res, field)
		for _, entry := range d {
			entry.FieldValues()[field] = BoolValue(g(entry) > middle)
		}
	}

	return res
}
